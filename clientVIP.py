import requests

def chat():
    user_name = input("Please enter your name: ")

    while True:
        user_input = input(f"{user_name}, what's on your mind (type 'nothing' to exit): ")
        if user_input.lower() == 'nothing':
            print("Exiting chat.")
            break

        # Send message to the server
        response = requests.post('http://localhost:5005/chat', json={"name": user_name, "message": user_input})

        if response.status_code == 200:
            data = response.json()
            print(f"Bot: {data['response']}")
        else:
            print("Error communicating with the server.")

if __name__ == '__main__':
    chat()
