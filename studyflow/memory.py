import hashlib
import json
import re
import sqlite3
from pathlib import Path

class Memory:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, name TEXT UNIQUE, version TEXT);
        CREATE TABLE IF NOT EXISTS sections(id INTEGER PRIMARY KEY, document_id INTEGER, version TEXT, heading TEXT, body TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(heading, body, content='sections', content_rowid='id');
        CREATE TABLE IF NOT EXISTS edges(a INTEGER, b INTEGER, PRIMARY KEY(a,b));
        CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, kind TEXT, body TEXT, stale INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS dependencies(artifact TEXT, document_id INTEGER, version TEXT);
        CREATE TABLE IF NOT EXISTS progress(run TEXT, question INTEGER, topic TEXT, correct INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(run,question));
        CREATE TABLE IF NOT EXISTS interactions(seq INTEGER PRIMARY KEY AUTOINCREMENT, run TEXT UNIQUE, thread TEXT NOT NULL, request TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, response TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if 'selected' not in {row['name'] for row in self.db.execute('PRAGMA table_info(progress)')}:
                self.db.execute('ALTER TABLE progress ADD COLUMN selected INTEGER')

    def close(self):
        self.db.close()

    def ingest(self, name, text):
        if Path(name).suffix.lower() not in {'.md', '.txt'}:
            raise ValueError('Поддерживаются только Markdown и TXT')
        if not text.strip() or len(text.encode()) > 1_000_000:
            raise ValueError('Нужен непустой UTF-8 файл размером до 1 МБ')
        if not any(line.strip() and not line.startswith('#') for line in text.splitlines()):
            raise ValueError('В файле только заголовки. Добавьте текст конспекта для поиска и вопросов.')
        version = hashlib.sha256(text.encode()).hexdigest()
        with self.db:
            old = self.db.execute('SELECT * FROM documents WHERE name=?', (name,)).fetchone()
            if old and old['version'] == version:
                return version
            if old:
                document_id = old['id']
                self.db.execute('UPDATE artifacts SET stale=1 WHERE id IN (SELECT artifact FROM dependencies WHERE document_id=?)', (document_id,))
                rows = self.db.execute('SELECT * FROM sections WHERE document_id=?', (document_id,)).fetchall()
                for row in rows:
                    self.db.execute("INSERT INTO search(search,rowid,heading,body) VALUES('delete',?,?,?)", (row['id'], row['heading'], row['body']))
                    self.db.execute('DELETE FROM edges WHERE a=? OR b=?', (row['id'], row['id']))
                self.db.execute('DELETE FROM sections WHERE document_id=?', (document_id,))
                self.db.execute('UPDATE documents SET version=? WHERE id=?', (version, document_id))
            else:
                document_id = self.db.execute('INSERT INTO documents(name,version) VALUES(?,?)', (name, version)).lastrowid
            heading, body, previous_section_id = 'Введение', [], None

            def save_section():
                nonlocal previous_section_id
                content = '\n'.join(body).strip()
                if not content:
                    return
                for offset in range(0, len(content), 1800):
                    title = f'{heading} [{offset // 1800 + 1}]'
                    chunk = content[offset:offset + 1800]
                    section_id = self.db.execute('INSERT INTO sections(document_id,version,heading,body) VALUES(?,?,?,?)', (document_id, version, title, chunk)).lastrowid
                    self.db.execute('INSERT INTO search(rowid,heading,body) VALUES(?,?,?)', (section_id, title, chunk))
                    if previous_section_id is not None:
                        self.db.execute('INSERT OR IGNORE INTO edges VALUES(?,?)', (previous_section_id, section_id))
                    previous_section_id = section_id
            for line in text.splitlines():
                if line.startswith('#'):
                    save_section()
                    heading, body = line.lstrip('#').strip(), []
                else:
                    body.append(line)
            save_section()
        return version

    def documents(self):
        return [dict(row) for row in self.db.execute('SELECT * FROM documents')]

    def recent_sections(self, limit=12):
        rows = self.db.execute(
            'SELECT * FROM (SELECT * FROM sections ORDER BY id DESC LIMIT ?) ORDER BY id',
            (max(1, min(int(limit), 24)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query):
        words = re.findall(r'\w+', query.lower())[:20]
        if not words:
            return []
        match = ' OR '.join('"' + w + '"*' for w in words)
        hits = self.db.execute('SELECT s.* FROM search JOIN sections s ON s.id=search.rowid WHERE search MATCH ? ORDER BY rank LIMIT 4', (match,)).fetchall()
        found_by_id = {row['id']: dict(row) for row in hits}
        for section_id in list(found_by_id):
            neighbors = self.db.execute(
                'SELECT s.* FROM edges e JOIN sections s ON s.id=CASE WHEN e.a=? THEN e.b ELSE e.a END WHERE e.a=? OR e.b=? LIMIT 1',
                (section_id, section_id, section_id),
            )
            for neighbor in neighbors:
                found_by_id[neighbor['id']] = dict(neighbor)
        return list(found_by_id.values())[:6]

    def progress(self):
        return [dict(row) for row in self.db.execute('SELECT topic, count(*) attempts, avg(correct) mastery FROM progress GROUP BY topic')]

    def save(self, run, kind, draft):
        with self.db:
            for citation in draft.citations:
                row = self.db.execute('SELECT 1 FROM sections WHERE id=? AND version=? AND heading=?', (citation.source_id, citation.version, citation.section)).fetchone()
                if not row:
                    raise ValueError('Источник изменился. Повторите запрос.')
            self.db.execute('INSERT INTO artifacts(id,kind,body) VALUES(?,?,?)', (run, kind, draft.model_dump_json()))
            for citation in draft.citations:
                self.db.execute('INSERT INTO dependencies SELECT ?,document_id,version FROM sections WHERE id=?', (run, citation.source_id))

    def artifact(self, run):
        row = self.db.execute('SELECT * FROM artifacts WHERE id=?', (run,)).fetchone()
        return dict(row) if row else None

    def artifact_sources(self, run):
        artifact = self.artifact(run)
        if not artifact or artifact['stale']:
            raise ValueError('Материал отсутствует или устарел')
        sources = []
        for citation in json.loads(artifact['body'])['citations']:
            row = self.db.execute('SELECT * FROM sections WHERE id=? AND version=? AND heading=?',
                                  (citation['source_id'], citation['version'], citation['section'])).fetchone()
            if not row:
                raise ValueError('Источник изменился. Повторите запрос.')
            sources.append(dict(row))
        return sources

    def record_interaction(self, result, request, thread='student'):
        draft = (result.get('draft') or {}) if result.get('status') == 'approved' else {'issues': result.get('verdict', {}).get('issues', [])}
        with self.db:
            self.db.execute(
                'INSERT INTO interactions(run,thread,request,kind,status,response) VALUES(?,?,?,?,?,?)',
                (result['run'], thread, request, result.get('mode', 'explain'), result.get('status', 'error'), json.dumps(draft, ensure_ascii=False)),
            )

    def history(self, thread='student', limit=50):
        rows = self.db.execute(
            '''SELECT i.run,i.request,i.kind,i.status,i.response,i.created,COALESCE(a.stale,0) stale
               FROM interactions i LEFT JOIN artifacts a ON a.id=i.run
               WHERE i.thread=? ORDER BY i.seq DESC LIMIT ?''',
            (thread, max(1, min(int(limit), 100))),
        ).fetchall()
        result = [dict(row) for row in rows]
        legacy = self.db.execute(
            '''SELECT a.id run,'' request,a.kind,'approved' status,a.body response,'' created,a.stale
               FROM artifacts a LEFT JOIN interactions i ON i.run=a.id
               WHERE i.run IS NULL ORDER BY a.rowid DESC LIMIT ?''',
            (max(1, min(int(limit), 100)),),
        ).fetchall()
        result.extend(dict(row) for row in legacy)
        return result[:limit]

    def grade(self, run, answers):
        with self.db:
            artifact = self.artifact(run)
            if not artifact or artifact['stale'] or artifact['kind'] != 'quiz':
                raise ValueError('Тест отсутствует или устарел')
            questions = json.loads(artifact['body'])['questions']
            if len(answers) != len(questions) or any(type(answer) is not int or not 0 <= answer < len(question['options']) for answer, question in zip(answers, questions)):
                raise ValueError('Ответьте на каждый вопрос')
            for index, (answer, question) in enumerate(zip(answers, questions)):
                self.db.execute('INSERT OR IGNORE INTO progress(run,question,topic,correct,selected) VALUES(?,?,?,?,?)', (run, index, question['topic'], int(answer == question['correct']), answer))
        return self.grades(run)

    def grades(self, run):
        return [dict(row) for row in self.db.execute('SELECT * FROM progress WHERE run=? ORDER BY question', (run,))]
