from app.main import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8012)
    # app.run(debug=True, host="192.168.6.59", port=8012)
