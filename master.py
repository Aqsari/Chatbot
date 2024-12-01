# ====================================
# SERVER CHATBOT EMOSIONAL
# ====================================
# Dibuat oleh: [Nama Anda]
# Versi: 1.0
# Deskripsi: Server untuk menangani komunikasi chat emosional

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from typing import Dict, List, Set
import json

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)


# Struktur data untuk menyimpan informasi
class ChatDatabase:
    def __init__(self):
        self.active_users: Dict[str, Dict] = {}
        self.messages: List[Dict] = []
        self.group_chat_users: Set[str] = set()
        self.emotion_history: Dict[str, List] = {}  # Menyimpan history emosi per user


database = ChatDatabase()


# ====================================
# FUNGSI UTILITAS
# ====================================

def log_activity(activity: str, user: str = "System") -> None:
    """
    Mencatat aktivitas server

    Parameters:
        activity (str): Aktivitas yang dicatat
        user (str): Pengguna yang melakukan aktivitas
    """
    logging.info(f"{user}: {activity}")


def validate_request_data(data: Dict, required_fields: List[str]) -> bool:
    """
    Memvalidasi data request

    Parameters:
        data (Dict): Data yang akan divalidasi
        required_fields (List[str]): Daftar field yang wajib ada

    Returns:
        bool: Status validasi
    """
    return all(field in data for field in required_fields)


# ====================================
# ROUTE API
# ====================================

@app.route('/api/join', methods=['POST'])
def join_chat():
    """
    Handler untuk bergabung ke chat
    """
    data = request.json

    if not validate_request_data(data, ['username']):
        return jsonify({
            'status': 'error',
            'message': 'Username diperlukan'
        }), 400

    username = data['username']
    chat_type = data.get('chat_type', 'Group')

    # Update data pengguna
    database.active_users[username] = {
        'last_seen': datetime.now(),
        'chat_type': chat_type,
        'emotion_state': 'netral'  # Status emosi awal
    }

    if chat_type == 'Group':
        database.group_chat_users.add(username)
        join_message = f"SISTEM: {username} bergabung ke dalam grup chat"
        database.messages.append({
            'message': join_message,
            'timestamp': datetime.now().isoformat(),
            'chat_type': 'Group',
            'emotion': 'system'
        })

        log_activity(f"{username} bergabung ke grup chat")

    return jsonify({
        'status': 'success',
        'message': f'Selamat datang {username}!',
        'active_users': list(database.group_chat_users)
    })


@app.route('/api/leave', methods=['POST'])
def leave_chat():
    """
    Handler untuk keluar dari chat
    """
    data = request.json

    if not validate_request_data(data, ['username']):
        return jsonify({
            'status': 'error',
            'message': 'Username diperlukan'
        }), 400

    username = data['username']
    chat_type = data.get('chat_type', 'Group')

    if username in database.active_users:
        del database.active_users[username]

        if chat_type == 'Group':
            database.group_chat_users.remove(username)
            leave_message = f"SISTEM: {username} meninggalkan grup chat"
            database.messages.append({
                'message': leave_message,
                'timestamp': datetime.now().isoformat(),
                'chat_type': 'Group',
                'emotion': 'system'
            })

            log_activity(f"{username} keluar dari grup chat")

    return jsonify({
        'status': 'success',
        'message': f'Sampai jumpa {username}!'
    })


@app.route('/api/message', methods=['POST'])
def receive_message():
    """
    Handler untuk menerima pesan
    """
    data = request.json

    if not validate_request_data(data, ['message']):
        return jsonify({
            'status': 'error',
            'message': 'Pesan diperlukan'
        }), 400

    message = data['message']
    chat_type = data.get('chat_type', 'Group')
    emotion = data.get('emotion', 'netral')

    # Simpan pesan dengan informasi tambahan
    message_data = {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'chat_type': chat_type,
        'emotion': emotion
    }
    database.messages.append(message_data)

    log_activity(f"Pesan baru diterima: {message[:50]}...")

    return jsonify({
        'status': 'success',
        'message': message
    })


@app.route('/api/messages', methods=['GET'])
def get_messages():
    """
    Handler untuk mengambil pesan
    """
    chat_type = request.args.get('chat_type', 'Group')
    username = request.args.get('username', '')

    # Filter pesan sesuai tipe chat
    if chat_type == 'Group':
        filtered_messages = [
            msg for msg in database.messages
            if msg['chat_type'] == 'Group'
        ]
    else:
        filtered_messages = [
            msg for msg in database.messages
            if msg['chat_type'].startswith('Personal')
               and username in msg['chat_type']
        ]

    return jsonify({
        'messages': filtered_messages,
        'active_users': list(database.group_chat_users)
    })


@app.route('/api/emotions/<username>', methods=['GET'])
def get_user_emotions(username):
    """
    Handler untuk mengambil history emosi pengguna
    """
    if username in database.emotion_history:
        return jsonify({
            'status': 'success',
            'emotions': database.emotion_history[username]
        })
    return jsonify({
        'status': 'error',
        'message': 'Riwayat emosi tidak ditemukan'
    }), 404


@app.route('/api/active_users', methods=['GET'])
def get_active_users():
    """
    Handler untuk mendapatkan daftar pengguna aktif
    """
    return jsonify({
        'active_users': list(database.group_chat_users),
        'total_users': len(database.active_users)
    })


@app.route('/api/user/status', methods=['POST'])
def update_user_status():
    """
    Handler untuk memperbarui status pengguna
    """
    data = request.json
    if not validate_request_data(data, ['username', 'status']):
        return jsonify({
            'status': 'error',
            'message': 'Username dan status diperlukan'
        }), 400

    username = data['username']
    status = data['status']

    if username in database.active_users:
        database.active_users[username]['status'] = status
        return jsonify({
            'status': 'success',
            'message': f'Status {username} diperbarui'
        })
    return jsonify({
        'status': 'error',
        'message': 'Pengguna tidak ditemukan'
    }), 404


# ====================================
# KONFIGURASI SERVER
# ====================================

def setup_server():
    """
    Melakukan setup awal server
    """
    log_activity("Server chat emosional dimulai")
    # Tambahkan konfigurasi tambahan di sini jika diperlukan


if __name__ == '__main__':
    setup_server()
    # Jalankan server dengan konfigurasi yang aman
    app.run(
        debug=True,  # Matikan saat production
        host='0.0.0.0',
        port=5002,
        threaded=True  # Mendukung multiple connections
    )