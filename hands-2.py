import cv2
import mediapipe as mp
import numpy as np

# --- Configurações ---
BRUSH_THICKNESS = 15
DRAW_COLOR = (255, 0, 255) # Magenta (formato BGR)
ERASER_COLOR = (0, 0, 0)   # Preto (para "apagar" no canvas preto)

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Aumentamos a confiança para evitar traços tremidos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# --- Configuração do Canvas de Desenho ---
# Lemos um frame inicial para saber o tamanho da tela
success, frame = cap.read()
h, w, c = frame.shape
# Criamos uma imagem preta (matriz de zeros) do mesmo tamanho do vídeo
imgCanvas = np.zeros((h, w, 3), np.uint8)

# Variáveis para guardar a posição anterior do dedo (para traçar a linha)
px, py = 0, 0

print("Iniciando V2: Pintura Virtual. Use o dedo indicador para desenhar.")
print("Pressione 'q' para sair e 'c' para limpar a tela.")

while True:
    success, frame = cap.read()
    if not success: break

    # Espelhar a imagem para ficar natural (importante para desenhar)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inferência
    results = hands.process(frame_rgb)

    # Se detectou mão
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Opcional: desenhar o esqueleto da mão para debug
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Lógica de Desenho ---
            # Obter o landmark 8 (Ponta do Dedo Indicador)
            landmark_8 = hand_landmarks.landmark[8]
            
            # Converter coordenadas normalizadas (0.0-1.0) para pixels reais da tela
            cx, cy = int(landmark_8.x * w), int(landmark_8.y * h)

            # Desenhar uma "mira" na ponta do dedo no frame original
            cv2.circle(frame, (cx, cy), 10, DRAW_COLOR, cv2.FILLED)

            # Se for o primeiro ponto detectado após perder a mão de vista
            if px == 0 and py == 0:
                 px, py = cx, cy

            # Desenhar uma linha no CANVAS entre o ponto anterior e o atual
            # Usamos o canvas para que o desenho persista
            cv2.line(imgCanvas, (px, py), (cx, cy), DRAW_COLOR, BRUSH_THICKNESS)
            
            # Atualizar o ponto anterior para o próximo loop
            px, py = cx, cy

    else:
        # Se perdeu a mão de vista, reseta os pontos anteriores para não traçar uma linha reta quando a mão voltar
        px, py = 0, 0

    # --- Fusão das Imagens ---
    # A mágica acontece aqui. Somamos o vídeo ao vivo com o canvas de desenho.
    # Como o canvas é preto (zero), onde não há desenho, a imagem da câmera não é alterada.
    
    # 1. Criar uma máscara da imagem desenhada (converte para escala de cinza)
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # 2. Inverter a máscara (tudo que era preto vira branco e vice-versa)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # 3. Onde o canvas tem desenho, deixamos o frame original preto nessa área
    frame = cv2.bitwise_and(frame, frame, mask=imgInv)
    # 4. Somamos o frame (agora com "buracos" pretos onde tem desenho) com o canvas colorido
    frame = cv2.bitwise_or(frame, imgCanvas)


    cv2.imshow('Visão Computacional V2 - Pintura Virtual AR', frame)

    # Controles do teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # Limpar o canvas
        imgCanvas = np.zeros((h, w, 3), np.uint8)
        print("Canvas limpo!")

cap.release()
cv2.destroyAllWindows()