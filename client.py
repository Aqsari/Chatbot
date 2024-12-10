# ====================================
# APLIKASI CHATBOT EMOSIONAL
# ====================================
# Dibuat oleh: [Nama Anda]
# Versi: 1.0
# Deskripsi: Sistem chat yang mendukung komunikasi personal dan grup
# dengan kemampuan deteksi emosi

import requests
import threading
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Konfigurasi logging untuk debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EmotionalChatClient:
    """
    Kelas utama untuk client chat dengan kemampuan deteksi emosi
    """

    def __init__(self, server_url: str = 'http://localhost:5002'):
        """Inisialisasi client chat

        Parameters:
            server_url (str): URL server chat
        """
        self.server_url = server_url
        self.username: Optional[str] = "ChatBot"  # Menetapkan username bot
        self.chat_type: Optional[str] = None
        self.last_message_time: Optional[str] = None
        self.running: bool = False
        self.message_thread: Optional[threading.Thread] = None
        self.emotion_history: List[Dict] = []  # Untuk menyimpan riwayat emosi

    def detect_emotion(self, message: str) -> str:
        """
        Mendeteksi emosi dari pesan

        Parameters:
            message (str): Pesan yang akan dianalisis

        Returns:
            str: Emosi yang terdeteksi (contoh: 'senang', 'sedih', 'marah')
        """
        # TODO: Implementasi deteksi emosi menggunakan model ML
        # Untuk saat ini return default
        return 'netral'

    def join_chat(self, username: str, chat_type: str) -> bool:
        """
        Bergabung ke dalam chat

        Parameters:
            username (str): Nama pengguna
            chat_type (str): Tipe chat ('Group' atau 'Personal')

        Returns:
            bool: Status keberhasilan bergabung
        """
        try:
            response = requests.post(
                f'{self.server_url}/api/join',
                json={'username': username, 'chat_type': chat_type},
                timeout=5  # Tambahan timeout untuk mencegah hanging
            )

            if response.status_code == 200:
                data = response.json()
                logging.info(f"Berhasil bergabung: {data['message']}")

                if chat_type == 'Group':
                    active_users = data.get('active_users', [])
                    print(f"\nPengguna aktif dalam grup: {', '.join(active_users)}")
                return True

            logging.error(f"Gagal bergabung: {response.status_code}")
            return False

        except requests.exceptions.RequestException as e:
            logging.error(f"Error saat bergabung chat: {e}")
            print(f"\nError: Tidak dapat terhubung ke server. {e}")
            return False

    def leave_chat(self):
        """
        Keluar dari chat
        """
        if self.username and self.chat_type:
            try:
                response = requests.post(
                    f'{self.server_url}/api/leave',
                    json={'username': self.username, 'chat_type': self.chat_type},
                    timeout=5
                )
                if response.status_code == 200:
                    logging.info(f"Berhasil keluar dari chat: {self.username}")
                else:
                    logging.error(f"Gagal keluar dari chat: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error saat keluar chat: {e}")
                print(f"\nError: Tidak dapat terhubung ke server. {e}")

    def fetch_messages(self) -> (List[Dict], List[str]):
        """
        Mengambil pesan dari server

        Returns:
            tuple: (messages, active_users)
        """
        try:
            response = requests.get(
                f'{self.server_url}/api/messages',
                params={'chat_type': self.chat_type, 'username': self.username},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('messages', []), data.get('active_users', [])
        except requests.exceptions.RequestException as e:
            logging.error(f"Error mengambil pesan: {e}")
        return [], []

    def message_polling(self):
        """
        Polling pesan dari server secara berkala
        """
        while self.running:
            try:
                messages, active_users = self.fetch_messages()
                if messages:
                    new_messages = [
                        msg for msg in messages
                        if not self.last_message_time or msg['timestamp'] > self.last_message_time
                    ]

                    for msg in new_messages:
                        print(f"\n{msg['message']}")

                    if new_messages:
                        self.last_message_time = messages[-1]['timestamp']

                time.sleep(1)  # Poll setiap detik
            except Exception as e:
                logging.error(f"Error dalam polling pesan: {e}")
                time.sleep(1)  # Tetap wait jika terjadi error

    def send_message(self, message: str) -> None:
        """
        Mengirim pesan dengan deteksi emosi

        Parameters:
            message (str): Pesan yang akan dikirim
        """
        # try:
        # Deteksi emosi dari pesan
        emotion = self.detect_emotion(message)
        self.emotion_history.append({
            'message': message,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })

        # the input have diffendt value with expeted format. 
        
        formatted_message = f"{self.username} ({self.chat_type}) [{emotion}]: {message}"
        print("here i am "+formatted_message)
        response = requests.post(
            f'{self.server_url}/api/chat',
            json={
                'message': formatted_message,
                'chat_type': self.chat_type,
                'emotion': emotion
            },
            timeout=5
        )

        if response.status_code != 200:
            logging.error(f"Error mengirim pesan: {response.json()}")
            print("\nError: Pesan tidak terkirim")

        # Tampilkan balasan dari bot berdasarkan emosi
        response_message = self.generate_response_based_on_emotion(emotion)
        print(f"Bot: {response_message}")

        # except requests.exceptions.RequestException as e:
            # logging.error(f"Error mengirim pesan: {e}")
            # print(f"\nError: Gagal mengirim pesan. {e}")

    def generate_response_based_on_emotion(self, emotion: str) -> str:
        """Menghasilkan balasan berdasarkan emosi yang terdeteksi."""
        if emotion == 'senang':
            return "Saya senang mendengar itu! Apa yang membuat Anda bahagia?"
        elif emotion == 'sedih':
            return "Saya minta maaf mendengar bahwa Anda merasa sedih. Apakah ada yang ingin Anda bicarakan?"
        elif emotion == 'marah':
            return "Saya mengerti bahwa Anda mungkin merasa marah. Mari kita coba tenangkan diri."
        else:
            return "Terima kasih telah berbagi. Bagaimana saya bisa membantu Anda lebih lanjut?"

    def start_chat(self, chat_type: str) -> None:
        """
        Memulai sesi chat

        Parameters:
            chat_type (str): Tipe chat yang akan dimulai
        """
        self.chat_type = chat_type
        if self.join_chat(self.username, chat_type):
            self.running = True
            self.message_thread = threading.Thread(target=self.message_polling)
            self.message_thread.daemon = True
            self.message_thread.start()

            print(f"\n{'=' * 50}")
            print(f"Selamat datang di ruang chat {chat_type}")
            print("Ketik 'quit' untuk keluar")
            print(f"{'=' * 50}\n")

            try:
                while True:
                    message = input()
                    if message.lower() == 'quit':
                        break
                    self.send_message(message)
            except KeyboardInterrupt:
                print("\nMenerima sinyal untuk keluar...")
            finally:
                self.cleanup()

    def cleanup(self) -> None:
        """
        Membersihkan resource saat keluar dari chat
        """
        self.running = False
        self.leave_chat()
        if self.message_thread:
            self.message_thread.join(timeout=1)
        print(f"\nBerhasil keluar dari ruang chat {self.chat_type}.")

def main():
    """
    Fungsi utama untuk menjalankan aplikasi
    """
    client = EmotionalChatClient()
    print("\n=== SELAMAT DATANG DI APLIKASI CHAT EMOSIONAL ===")

    try:
        username = input("\nMasukkan nama pengguna Anda: ").strip()
        while not username:
            print("Error: Nama pengguna tidak boleh kosong!")
            username = input("Masukkan nama pengguna Anda: ").strip()

        client.username = username

        while True:
            print("\nPilihan Menu:")
            print("1. Bergabung ke chat grup")
            print("2. Mulai chat personal")
            print("3. Keluar")

            choice = input("\nPilih menu (1-3): ")

            if choice == '1':
                client.start_chat("Group")
            elif choice == '2':
                recipient = input("Masukkan username tujuan: ").strip()
                while not recipient:
                    print("Error: Username tujuan tidak boleh kosong!")
                    recipient = input("Masukkan username tujuan: ").strip()
                client.start_chat(f"Personal with {recipient}")
            elif choice == '3':
                print("\nTerima kasih telah menggunakan aplikasi chat!")
                break
            else:
                print("\nError: Pilihan tidak valid. Silakan pilih 1, 2, atau 3.")

    except KeyboardInterrupt:
        print("\n\nMenerima permintaan keluar. Menutup aplikasi...")
    except Exception as e:
        logging.error(f"Error tidak terduga: {e}")
        print("\nTerjadi error yang tidak terduga. Aplikasi akan ditutup.")
    finally:
        if hasattr(client, 'cleanup'):
            client.cleanup()

if __name__ == '__main__':
    main()
