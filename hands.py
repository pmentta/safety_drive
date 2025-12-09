import sys
import cv2
import mediapipe as mp

print("sys.executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)

# 1. Configuração do MediaPipe (o "cérebro" da visão computacional)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Utilitário para desenhar os ossos

# Inicializa o modelo
# static_image_mode=False: Otimiza para vídeo (usa o frame anterior para agilizar o próximo)
# max_num_hands=2: Detecta até duas mãos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# 2. Captura da Webcam
cap = cv2.VideoCapture(0) # '0' geralmente é a webcam padrão

print("Iniciando... Pressione 'q' na janela do vídeo para sair.")

while True:
    # Ler o frame da câmera
    success, frame = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # 3. Pré-processamento
    # O OpenCV usa BGR, mas o MediaPipe precisa de RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Processamento (Inferência)
    results = hands.process(frame_rgb)

    # 5. Desenhando os resultados
    # Se houver mãos detectadas (results.multi_hand_landmarks não é None)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos (nós) e as conexões (ossos) na imagem original
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), # Cor dos pontos
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)  # Cor das linhas
            )

    # Mostrar o resultado final numa janela
    # Espelhamos a imagem (flip) para ficar mais natural como um espelho
    cv2.imshow('Visão Computacional - Hand Tracking', cv2.flip(frame, 1))

    # Pressione 'q' para sair do loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Limpeza final
cap.release()
cv2.destroyAllWindows()