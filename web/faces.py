"""Face detection + recognition with persistent per-person profiles.

Uses face_recognition (dlib) for detection + 128-d embeddings and Chroma for
the per-person vector store. Each entry carries a JSON profile (name, notes,
first_seen, last_seen, greeting_count).
"""

import io
import json
import time
import threading
from pathlib import Path
from typing import Optional


KNOWN_GREET_COOLDOWN_SEC = 300.0    # don't auto-greet the same known person within this window
UNKNOWN_GREET_COOLDOWN_SEC = 120.0  # don't keep asking who an unknown face is too often
MATCH_SIMILARITY_THRESHOLD = 0.85   # cosine sim threshold for "this is the same person"


class FaceSystem:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.fr = None
        self.np = None
        self.chroma = None
        self.collection = None
        self.available = False

        self.last_frame: Optional[bytes] = None
        # Per-snapshot face list: [{bbox, encoding, match}]
        self.last_faces: list[dict] = []
        self.last_detect_ts: float = 0.0
        self._lock = threading.Lock()

        # Greet bookkeeping.
        self._last_greeted_known: dict[str, float] = {}  # person_id -> ts
        self._last_asked_unknown: float = 0.0

        try:
            import numpy as np
            import chromadb
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            print(f"Face deps not installed: {exc}")
            print("Install with: pip install insightface onnxruntime chromadb numpy pillow")
            return

        try:
            self.np = np
            print("[faces] loading InsightFace buffalo_l (downloads ~280MB on first run)…")
            self.fr = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
                root=str(self.models_dir / "insightface"),
            )
            self.fr.prepare(ctx_id=0, det_size=(640, 640))
            self.chroma = chromadb.PersistentClient(path=str(self.models_dir / "chroma_faces"))
            self.collection = self.chroma.get_or_create_collection(
                name="people", metadata={"hnsw:space": "cosine"}
            )
            self.available = True
            print(f"[faces] ready ({self.collection.count()} people)")
        except Exception as exc:
            print(f"[faces] init failed: {exc}")

    # ---- core ----

    def detect(self, image_bytes: bytes) -> list[dict]:
        if not self.available:
            return []
        from PIL import Image

        rgb = self.np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()  # insightface expects BGR like cv2
        results = self.fr.get(bgr)
        faces = []
        for r in results:
            bbox = [float(v) for v in r.bbox.tolist()]  # x1, y1, x2, y2
            emb = r.normed_embedding if hasattr(r, "normed_embedding") else r.embedding
            # Ensure L2-normalized for cosine similarity.
            emb = self.np.asarray(emb, dtype=self.np.float32)
            norm = self.np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            enc_list = emb.tolist()
            match = self._match(emb)
            faces.append(
                {
                    "bbox": bbox,
                    "encoding": enc_list,
                    "match": match,
                    "det_score": float(getattr(r, "det_score", 0.0)),
                }
            )
            if match and match["similarity"] >= MATCH_SIMILARITY_THRESHOLD:
                self._touch_last_seen(match["id"])
        with self._lock:
            self.last_faces = faces
            self.last_frame = image_bytes
            self.last_detect_ts = time.time()
        if faces:
            summary = []
            for f in faces:
                m = f.get("match")
                if m and m["similarity"] >= MATCH_SIMILARITY_THRESHOLD:
                    summary.append(f"{m['name']}({m['similarity']:.2f})")
                elif m:
                    summary.append(f"?(closest:{m['name']}@{m['similarity']:.2f})")
                else:
                    summary.append("unknown")
            print(f"[faces] detect: {len(faces)} face(s) → {', '.join(summary)}", flush=True)
        else:
            # Throttle to once every ~6s so the log isn't spammy when nobody's there.
            now = time.time()
            if not hasattr(self, "_last_empty_log") or now - self._last_empty_log > 6:
                self._last_empty_log = now
                print("[faces] detect: 0 faces", flush=True)
        return faces

    def _match(self, enc) -> Optional[dict]:
        if self.collection.count() == 0:
            return None
        enc_list = enc.tolist() if hasattr(enc, "tolist") else list(enc)
        q = self.collection.query(query_embeddings=[enc_list], n_results=1)
        ids = q["ids"][0] if q["ids"] else []
        if not ids:
            return None
        sim = max(0.0, 1.0 - float(q["distances"][0][0]))
        meta = q["metadatas"][0][0]
        if sim < MATCH_SIMILARITY_THRESHOLD - 0.10:  # don't even report poor matches
            return None
        return {
            "id": ids[0],
            "name": meta.get("name", "?"),
            "similarity": sim,
            "profile": self._load_profile(meta),
            "first_seen": meta.get("first_seen"),
            "last_seen": meta.get("last_seen"),
            "greeting_count": meta.get("greeting_count", 0),
        }

    def _touch_last_seen(self, person_id: str):
        existing = self.collection.get(ids=[person_id])
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0])
        meta["last_seen"] = time.time()
        self.collection.update(ids=[person_id], metadatas=[meta])

    # ---- enrollment / profile editing ----

    PROFILE_LIST_FIELDS = ("interests", "allergies", "dietary", "preferences", "topics_to_avoid")

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "interests": [],
            "allergies": [],
            "dietary": [],
            "preferences": [],
            "topics_to_avoid": [],
            "notes": "",
        }

    def _load_profile(self, meta: dict) -> dict:
        raw = meta.get("profile")
        if not raw:
            # legacy migration: old rows only had `notes`
            base = self._empty_profile()
            base["notes"] = meta.get("notes", "") or ""
            return base
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        merged = self._empty_profile()
        for k in merged:
            v = data.get(k)
            if v is not None:
                merged[k] = v
        return merged

    def _save_profile(self, person_id: str, meta: dict, profile: dict) -> None:
        new_meta = dict(meta)
        new_meta["profile"] = json.dumps(profile, ensure_ascii=False)
        self.collection.update(ids=[person_id], metadatas=[new_meta])

    def enroll_from_last_frame(self, name: str, notes: str = "") -> Optional[dict]:
        """Save the largest face from the most recent detection under `name`."""
        if not self.available:
            return None
        with self._lock:
            faces = list(self.last_faces)
        if not faces:
            return None

        def area(f):
            x1, y1, x2, y2 = f["bbox"]  # insightface: (x1, y1, x2, y2)
            return abs((x2 - x1) * (y2 - y1))

        face = max(faces, key=area)
        enc = face["encoding"]
        now = time.time()
        item_id = f"person_{int(now * 1000)}"
        bbox_int = [int(v) for v in face["bbox"]]
        profile = self._empty_profile()
        if notes:
            profile["notes"] = notes
        self.collection.add(
            ids=[item_id],
            embeddings=[enc],
            metadatas=[
                {
                    "name": name,
                    "first_seen": now,
                    "last_seen": now,
                    "greeting_count": 0,
                    "profile": json.dumps(profile, ensure_ascii=False),
                }
            ],
        )
        print(f"[faces] ENROLLED id={item_id} name='{name}' bbox={bbox_int} "
              f"(total people: {self.collection.count()})", flush=True)
        return {"id": item_id, "name": name}

    def update_profile(self, person_id: str, updates: dict) -> Optional[dict]:
        """Merge structured updates into a person's profile.

        `updates` may include any of the list fields (`interests`, `allergies`,
        `dietary`, `preferences`, `topics_to_avoid`) — values are appended +
        deduped — and `notes` (string, appended to existing free-text).
        """
        if not self.available:
            return None
        existing = self.collection.get(ids=[person_id])
        if not existing["ids"]:
            print(f"[faces] update_profile skipped — unknown person_id={person_id}", flush=True)
            return None
        meta = dict(existing["metadatas"][0])
        profile = self._load_profile(meta)
        added: dict = {}
        for field in self.PROFILE_LIST_FIELDS:
            incoming = updates.get(field) or []
            if not incoming:
                continue
            current = list(profile.get(field) or [])
            current_lc = {c.lower() for c in current}
            new = [x.strip() for x in incoming if x and x.strip() and x.strip().lower() not in current_lc]
            if new:
                profile[field] = current + new
                added[field] = new
        note = (updates.get("notes") or "").strip()
        if note:
            existing_notes = profile.get("notes") or ""
            profile["notes"] = (existing_notes + "\n" + note).strip()
            added["notes"] = note
        if not added:
            return {"id": person_id, "name": meta.get("name"), "added": {}}
        self._save_profile(person_id, meta, profile)
        print(f"[faces] PROFILE updated for '{meta.get('name')}' (id={person_id}): {added}", flush=True)
        return {"id": person_id, "name": meta.get("name"), "added": added, "profile": profile}

    def append_note(self, person_id: str, note: str) -> bool:
        r = self.update_profile(person_id, {"notes": note})
        return bool(r and r.get("added", {}).get("notes"))

    def get_profile(self, person_id: str) -> Optional[dict]:
        if not self.available:
            return None
        existing = self.collection.get(ids=[person_id])
        if not existing["ids"]:
            return None
        meta = dict(existing["metadatas"][0])
        return {
            "id": person_id,
            "name": meta.get("name"),
            "first_seen": meta.get("first_seen"),
            "last_seen": meta.get("last_seen"),
            "profile": self._load_profile(meta),
        }

    def list_people(self) -> list[dict]:
        if not self.available:
            return []
        items = self.collection.get()
        out = []
        for i, m in zip(items.get("ids", []), items.get("metadatas", [])):
            out.append(
                {
                    "id": i,
                    "name": m.get("name"),
                    "profile": self._load_profile(m),
                    "first_seen": m.get("first_seen"),
                    "last_seen": m.get("last_seen"),
                }
            )
        return out

    def forget(self, person_id: str) -> bool:
        if not self.available:
            return False
        self.collection.delete(ids=[person_id])
        self._last_greeted_known.pop(person_id, None)
        print(f"[faces] FORGOT id={person_id} (remaining: {self.collection.count()})")
        return True

    # ---- trigger logic ----

    def pop_greeting_trigger(self) -> Optional[dict]:
        """Return a trigger dict (or None) if a person-driven check-in should fire.

        Sets the cooldown bookkeeping when it returns a trigger, so the caller
        doesn't need to. Returns None if no faces are visible.
        """
        if not self.available:
            return None
        with self._lock:
            faces = list(self.last_faces)
        if not faces:
            return None

        # Prefer a known, un-greeted person; fall back to unknown.
        known = [f for f in faces if f.get("match") and f["match"]["similarity"] >= MATCH_SIMILARITY_THRESHOLD]
        unknown = [f for f in faces if not (f.get("match") and f["match"]["similarity"] >= MATCH_SIMILARITY_THRESHOLD)]
        now = time.time()

        if known:
            for f in known:
                m = f["match"]
                last = self._last_greeted_known.get(m["id"], 0.0)
                if now - last >= KNOWN_GREET_COOLDOWN_SEC:
                    self._last_greeted_known[m["id"]] = now
                    print(f"[faces] TRIGGER: known_person {m['name']} (id={m['id']}, sim={m['similarity']:.2f})")
                    return {
                        "type": "known_person",
                        "person": {
                            "id": m["id"],
                            "name": m["name"],
                            "notes": m.get("notes", ""),
                            "first_seen": m.get("first_seen"),
                            "last_seen": m.get("last_seen"),
                            "similarity": m["similarity"],
                        },
                    }
        if unknown:
            if now - self._last_asked_unknown >= UNKNOWN_GREET_COOLDOWN_SEC:
                self._last_asked_unknown = now
                print(f"[faces] TRIGGER: unknown_person")
                return {"type": "unknown_person"}
        return None

    def scene_context_string(self) -> str:
        if not self.last_faces:
            return ""
        lines: list[str] = []
        for f in self.last_faces:
            m = f.get("match")
            if m and m["similarity"] >= MATCH_SIMILARITY_THRESHOLD:
                p = m.get("profile") or {}
                bits = [f"Person in view: {m['name']} (id={m['id']})."]
                for field in self.PROFILE_LIST_FIELDS:
                    vals = p.get(field) or []
                    if vals:
                        bits.append(f"{field}: {', '.join(vals)}.")
                notes = (p.get("notes") or "").strip()
                if notes:
                    bits.append(f"notes: {notes}")
                lines.append(" ".join(bits))
            else:
                lines.append("Person in view: unknown face.")
        return " ".join(lines)
