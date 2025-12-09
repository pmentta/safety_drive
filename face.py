import cv2
from fer.fer import FER
import time

# 1. Configuração do Detector
# Inicializa o detector de emoções.
# O padrão usa o detector de faces do OpenCV (Haar Cascade), que é rápido.
# Se quiser mais precisão (mas mais lento), use: detector = FER(mtcnn=True)
detector = FER()

cap = cv2.VideoCapture(0)

# Configurações de fonte e cor para o texto
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 255) # Amarelo (BGR)
thickness = 3
box_color = (0, 255, 0) # Verde para o quadrado do rosto

# Variáveis para calcular FPS (Frames Por Segundo)
pTime = 0
cTime = 0

print("Iniciando detector de emoções...")
print("A primeira execução pode demorar um pouco para baixar o modelo.")
print("Pressione 'q' para sair.")

while True:
    # Ler frame da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Opcional: Espelhar o frame
    frame = cv2.flip(frame, 1)

    # O detector funciona melhor (ou às vezes só funciona) com imagens RGB
    # O OpenCV lê em BGR, então convertemos.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Inferência (A mágica acontece aqui)
    # detector.detect_emotions retorna uma lista de dicionários.
    # Cada item tem a 'box' (caixa do rosto) e 'emotions' (as notas para cada emoção)
    analysis = detector.detect_emotions(frame_rgb)

    emotion_text = "Procurando rosto..."
    
    # Se detectou pelo menos um rosto
    if len(analysis) > 0:
        # Vamos pegar apenas o primeiro rosto detectado para simplificar
        first_face = analysis[0]
        
        # Extrair coordenadas do quadrado do rosto (x, y, largura, altura)
        (x, y, w, h) = first_face["box"]
        
        # Desenhar o quadrado no rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        
        # Extrair as emoções e encontrar a dominante
        emotions_dict = first_face["emotions"]
        # Acha a chave (emoção) com o maior valor
        dominant_emotion = max(emotions_dict, key=emotions_dict.get)
        score = emotions_dict[dominant_emotion]
        
        # Traduzindo para português para ficar mais legal
        translations = {
            "angry": "Raiva",
            "disgust": "Nojo",
            "fear": "Medo",
            "happy": "Feliz",
            "sad": "Triste",
            "surprise": "Surpresa",
            "neutral": "Neutro"
        }
        emotion_pt = translations.get(dominant_emotion, dominant_emotion)

        # Formata o texto final com a emoção e a confiança (ex: Feliz: 98%)
        emotion_text = f"{emotion_pt}: {int(score * 100)}%"

    # 3. Exibição no canto da tela
    # Desenhamos um fundo preto no canto superior esquerdo para o texto ficar legível
    cv2.rectangle(frame, (0,0), (400, 60), (0,0,0), cv2.FILLED)
    cv2.putText(frame, emotion_text, (10, 45), font, font_scale, font_color, thickness)

    # Cálculo de FPS (opcional, para ver a performance)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 120, 45), font, 0.8, (255,255,255), 2)


    cv2.imshow('Detector de Emocoes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()