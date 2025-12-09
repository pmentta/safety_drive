import cv2
import mediapipe as mp

# --- Configuração Inicial ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# max_num_hands=2 garante que funcione com ambas as mãos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# IDs das pontas dos dedos (4=Polegar, 8=Indicador, etc.)
tipIds = [4, 8, 12, 16, 20]

print("Iniciando... Mostre as duas mãos.")
print("Pressione 'q' para sair.")

while True:
    success, frame = cap.read()
    if not success: break

    # 1. Preparar a imagem
    frame = cv2.flip(frame, 1) # Espelhar para agir como um espelho
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Processar (Buscar mãos)
    results = hands.process(frame_rgb)

    # 3. Se achou mãos
    if results.multi_hand_landmarks:
        # O 'zip' permite loopar pelas landmarks (pontos) e pela classificação (esq/dir) ao mesmo tempo
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # --- A. Desenhar o Esqueleto ---
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            # --- B. Lógica de Contagem ---
            lm_list = []
            h, w, c = frame.shape
            
            # Converter coordenadas relativas (0.0-1.0) para pixels (x, y)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if lm_list:
                fingers = []
                
                # Identificar qual mão é (Left ou Right)
                # O MediaPipe inverte Labels quando usamos flip, então ajustamos:
                my_hand_type = hand_info.classification[0].label

                # --- Regra do Polegar (Lateral) ---
                # O polegar se move no eixo X (lados), não Y (altura)
                # Para a mão Direita na tela espelhada, polegar aberto é x da ponta < x da base
                if my_hand_type == "Right":
                    if lm_list[tipIds[0]][1] < lm_list[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else: # Mão Esquerda
                    if lm_list[tipIds[0]][1] > lm_list[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- Regra dos 4 Dedos (Vertical) ---
                # Ponta (id) deve estar acima (menor Y) que a segunda junta (id-2)
                for id in range(1, 5):
                    if lm_list[tipIds[id]][2] < lm_list[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- C. Mostrar Contagem ---
                total_fingers = fingers.count(1)
                
                # Posição do texto (em cima da mão)
                text_x = lm_list[0][1] - 30
                text_y = lm_list[0][2] + 50
                
                # Desenha um retângulo de fundo para o número ficar legível
                cv2.rectangle(frame, (text_x - 10, text_y - 40), (text_x + 60, text_y + 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(total_fingers), (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("Contador de Dedos Simples", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()