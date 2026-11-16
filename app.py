
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import ollama
import streamlit as st


# ---------------------------------------
# 1. Grundeinstellungen
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chats.db"

SYSTEM_PROMPT = (
    "Du bist der interne KI-Assistent der Lumora Energie GmbH. "
)

ASSISTANT_AVATAR = str(BASE_DIR / "images" / "robot.jpg")


# ---------------------------------------
# 2. SQLite-Datenbank
# ---------------------------------------

def connect():
    db = sqlite3.connect(DB_PATH, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=20000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    with connect() as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'done',
                info TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS msg_chat
            ON messages(chat_id, created_at)
        """)


def list_chats():
    with connect() as db:
        return db.execute(
            "SELECT * FROM chats ORDER BY created_at DESC"
        ).fetchall()


def get_messages(chat_id):
    with connect() as db:
        return db.execute(
            "SELECT * FROM messages WHERE chat_id=? "
            "ORDER BY created_at, rowid",
            (chat_id,)
        ).fetchall()


def create_chat():
    chat_id = str(uuid.uuid4())

    with connect() as db:
        db.execute(
            "INSERT INTO chats (id, title, created_at) "
            "VALUES (?, ?, ?)",
            (chat_id, "Neuer Chat", time.time())
        )

    return chat_id


def update_message(msg_id, content, status, info=""):
    with connect() as db:
        db.execute(
            "UPDATE messages "
            "SET content=?, status=?, info=? "
            "WHERE id=?",
            (content, status, info, msg_id)
        )


def has_running(chat_id):
    with connect() as db:
        return db.execute(
            "SELECT 1 FROM messages "
            "WHERE chat_id=? AND status='running' LIMIT 1",
            (chat_id,)
        ).fetchone() is not None


# ---------------------------------------
# 3. Hintergrundgenerierung
# ---------------------------------------

class TaskManager:

    def __init__(self):
        init_db()

        self.lock = threading.Lock()
        self.running = set()

        # Nach einem Neustart unterbrochene
        # Generierungen entsprechend markieren.
        with connect() as db:
            db.execute(
                "UPDATE messages "
                "SET status='interrupted', "
                "info='Generierung durch App-Neustart unterbrochen' "
                "WHERE status='running'"
            )

    def start(self, chat_id, prompt, model):

        with self.lock:

            if chat_id in self.running or has_running(chat_id):
                return False

            self.running.add(chat_id)

            try:
                now = time.time()
                answer_id = str(uuid.uuid4())

                with connect() as db:

                    existing = db.execute(
                        "SELECT COUNT(*) FROM messages "
                        "WHERE chat_id=?",
                        (chat_id,)
                    ).fetchone()[0]

                    if existing == 0:
                        title = prompt[:25] + (
                            "..." if len(prompt) > 25 else ""
                        )

                        db.execute(
                            "UPDATE chats SET title=? WHERE id=?",
                            (title, chat_id)
                        )

                    # Benutzernachricht speichern.
                    db.execute(
                        "INSERT INTO messages "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            chat_id,
                            "user",
                            prompt,
                            "done",
                            "",
                            now
                        )
                    )

                    # Leere KI-Antwort anlegen.
                    db.execute(
                        "INSERT INTO messages "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            answer_id,
                            chat_id,
                            "assistant",
                            "",
                            "running",
                            "",
                            now + 0.0001
                        )
                    )

                # Eigenen Hintergrundthread starten.
                thread = threading.Thread(
                    target=self._generate,
                    args=(chat_id, answer_id, model),
                    daemon=True
                )

                thread.start()

                return True

            except Exception:
                self.running.discard(chat_id)
                raise

    def _generate(self, chat_id, answer_id, model):

        text = ""
        start = time.monotonic()

        try:
            rows = get_messages(chat_id)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

            # Bisherigen Gesprächsverlauf zusammenstellen.
            for row in rows:

                if row["id"] == answer_id:
                    break

                if (
                    row["status"] == "done"
                    and row["role"] in ("user", "assistant")
                ):
                    messages.append({
                        "role": row["role"],
                        "content": row["content"]
                    })

            last_save = time.monotonic()

            # Ollama läuft im Hintergrund.
            for chunk in ollama.chat(
                model=model,
                messages=messages,
                stream=True
            ):

                text += chunk["message"]["content"]

                # Antwort regelmäßig speichern.
                if time.monotonic() - last_save >= 0.25:

                    update_message(
                        answer_id,
                        text,
                        "running"
                    )

                    last_save = time.monotonic()

            info = (
                f"⏱️ Antwortzeit "
                f"{time.monotonic() - start:.1f} s ({model})"
            )

            # Fertige Antwort speichern.
            update_message(
                answer_id,
                text,
                "done",
                info
            )

        except Exception as exc:

            update_message(
                answer_id,
                text,
                "error",
                f"Fehler: {exc}"
            )

        finally:

            with self.lock:
                self.running.discard(chat_id)


# ---------------------------------------
# 4. Streamlit starten
# ---------------------------------------

@st.cache_resource
def get_manager():
    return TaskManager()


st.set_page_config(
    page_title="Lumora On-Premise-KI",
    page_icon="🛡️"
)

manager = get_manager()


# Aktuellen Chat auswählen.
if "current_chat_id" not in st.session_state:

    chats = list_chats()

    st.session_state.current_chat_id = (
        chats[0]["id"] if chats else create_chat()
    )


# ---------------------------------------
# 5. Hauptüberschrift
# ---------------------------------------

st.title(
    "🛡️ Lumora On-Premise-KI",
    text_alignment="center"
)

st.caption(
    "Ihr interner KI-Assistent – lokal gespeichert",
    text_alignment="center"
)


# ---------------------------------------
# 6. Seitenleiste
# ---------------------------------------

with st.sidebar:

    if st.button(
        "Neuer Chat",
        use_container_width=True,
        icon=":material/add:"
    ):

        st.session_state.current_chat_id = create_chat()
        st.rerun()

    st.subheader("Bisherige Chats")

    for chat in list_chats():

        active = (
            chat["id"]
            == st.session_state.current_chat_id
        )

        if st.button(
            chat["title"],
            key=f"chat_{chat['id']}",
            type="primary" if active else "secondary",
            use_container_width=True
        ):

            st.session_state.current_chat_id = chat["id"]
            st.rerun()

    st.container(height=100, border=False)

    st.header("Einstellungen")

    modell = st.text_input(
        "Modell (llama-Version)",
        value="llama3.2:1b"
    )


chat_id = st.session_state.current_chat_id


# ---------------------------------------
# 7. Chatverlauf automatisch aktualisieren
# ---------------------------------------

@st.fragment(run_every="1s")
def chat_verlauf(selected_chat_id):

    for msg in get_messages(selected_chat_id):

        avatar = (
            ASSISTANT_AVATAR
            if msg["role"] == "assistant"
            else ":material/person:"
        )

        with st.chat_message(
            msg["role"],
            avatar=avatar
        ):

            if msg["content"]:

                st.markdown(
                    msg["content"]
                    + (
                        " ▌"
                        if msg["status"] == "running"
                        else ""
                    )
                )

            if msg["status"] == "running":

                st.caption(
                    "⏳ Antwort wird im Hintergrund generiert …"
                )

            elif msg["status"] == "interrupted":

                st.caption("⚠️ Generierung unterbrochen")

            elif msg["status"] == "error":

                st.caption("❌ Fehler bei der Generierung")

            if msg["info"]:
                st.caption(msg["info"])


chat_verlauf(chat_id)


if has_running(chat_id):

    st.caption(
        "Dieser Chat generiert gerade. "
        "Du kannst zu einem anderen Chat wechseln."
    )


# ---------------------------------------
# 8. Neue Nachricht verarbeiten
# ---------------------------------------

if prompt := st.chat_input("Schreib eine Nachricht..."):

    if manager.start(chat_id, prompt, modell):

        st.rerun()

    else:

        st.warning(
            "In diesem Chat wird bereits "
            "eine Antwort generiert."
        )
