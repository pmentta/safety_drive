import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os

# --- Tenta importar winsound para alertas sonoros (Apenas Windows) ---
try:
    import winsound
    def beep_alert():
        # Frequência 1000Hz, Duração 100ms (curto para não travar o loop)
        winsound.Beep(1000, 100)
except ImportError:
    def beep_alert():
        print("ALERTA SONORO! (winsound não disponível neste SO)")

# --- Configurações do Sistema ---
# Olhos
EAR_THRESHOLD = 0.25   # Se a abertura do olho for menor que isso, considera fechado
EYE_CLOSED_TIME_LIMIT = 1.5 # Segundos permitidos com olho fechado (piscada é ~0.3s)

# Mãos
HANDS_TIME_LIMIT = 2.0 # Segundos permitidos sem mãos no volante

# Cores (BGR)
COLOR_SAFE = (0, 255, 0)      # Verde
COLOR_WARNING = (0, 255, 255) # Amarelo
COLOR_DANGER = (0, 0, 255)    # Vermelho
COLOR_WHEEL = (200, 200, 200) # Cinza claro para o volante simulado

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# FaceMesh (refine_landmarks=True nos dá contornos de íris mais precisos se precisar)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Índices dos Landmarks dos Olhos (Topologia do MediaPipe) ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, indices, w, h):
    """Calcula o Eye Aspect Ratio (Razão de Aspecto do Olho)"""
    # Coordenadas dos pontos
    coords = []
    for i in indices:
        lm = landmarks[i]
        coords.append((int(lm.x * w), int(lm.y * h)))
    
    # Distâncias verticais (P2-P6 e P3-P5)
    v1 = math.dist(coords[1], coords[5])
    v2 = math.dist(coords[2], coords[4])
    
    # Distância horizontal (P1-P4)
    h_dist = math.dist(coords[0], coords[3])
    
    if h_dist == 0: return 0
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

def is_hand_in_zone(hand_landmarks, zone_rect):
    """Verifica se o pulso (0) e o dedo médio (9) estão na zona do volante"""
    # zone_rect = (x1, y1, x2, y2)
    x1, y1, x2, y2 = zone_rect
    
    # Verifica o pulso (landmark 0)
    wrist = hand_landmarks.landmark[0]
    
    # Convertendo para proporção (0.0 a 1.0) para comparar se estivéssemos usando pixels
    # Mas aqui vamos comparar direto se wrist.x e wrist.y estão "dentro" logicamente
    # O MediaPipe retorna x,y de 0 a 1.
    # Vamos assumir que zone_rect já está em coordenadas normalizadas (0 a 1)
    
    if x1 < wrist.x < x2 and y1 < wrist.y < y2:
        return True
    return False

# --- Loop Principal ---
cap = cv2.VideoCapture(0)

# Variáveis de Estado
start_time_eyes = None
start_time_hands = None
eyes_closed_duration = 0
hands_missing_duration = 0

print("Sistema de Monitoramento Iniciado. Pressione 'q' para sair.")

while True:
    success, frame = cap.read()
    if not success: break

    # Espelhar imagem
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Definir a Zona do Volante (Simulada)
    # Retângulo na parte inferior central (x_start, y_start, x_end, y_end) em PIXELS
    # Vamos dizer que o volante ocupa os 30% inferiores e 60% centrais da tela
    wheel_h = int(h * 0.35)
    wheel_y = h - wheel_h
    wheel_x = int(w * 0.2)
    wheel_w = int(w * 0.6)
    
    wheel_rect_pixels = (wheel_x, wheel_y, wheel_x + wheel_w, h) # x1, y1, x2, y2
    
    # Para verificação lógica (0.0 a 1.0)
    wheel_rect_norm = (0.2, 0.65, 0.8, 1.0) # x1, y1, x2, y2 (Y começa em 0 no topo)

    # Desenhar o "Volante" (Zona Segura)
    cv2.rectangle(frame, (wheel_x, wheel_y), (wheel_x + wheel_w, h), COLOR_WHEEL, 2)
    cv2.putText(frame, "AREA DO VOLANTE", (wheel_x + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHEEL, 2)

    # ==========================
    # PROCESSAMENTO DE FACE (SONO)
    # ==========================
    results_face = face_mesh.process(frame_rgb)
    drowsy_alert = False

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            lm_list = face_landmarks.landmark
            
            # Calcular EAR dos dois olhos
            ear_left = calculate_ear(lm_list, LEFT_EYE, w, h)
            ear_right = calculate_ear(lm_list, RIGHT_EYE, w, h)
            avg_ear = (ear_left + ear_right) / 2.0

            # Lógica de Tempo
            if avg_ear < EAR_THRESHOLD:
                if start_time_eyes is None:
                    start_time_eyes = time.time()
                
                eyes_closed_duration = time.time() - start_time_eyes
                
                # Barra de Progresso do Sono
                bar_width = 300
                fill = min(1.0, eyes_closed_duration / EYE_CLOSED_TIME_LIMIT)
                cv2.rectangle(frame, (50, 50), (50 + bar_width, 80), (50, 50, 50), cv2.FILLED)
                cv2.rectangle(frame, (50, 50), (50 + int(bar_width * fill), 80), COLOR_DANGER, cv2.FILLED)
                cv2.putText(frame, f"OLHOS FECHADOS: {eyes_closed_duration:.1f}s", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DANGER, 2)

                if eyes_closed_duration > EYE_CLOSED_TIME_LIMIT:
                    drowsy_alert = True
                    cv2.putText(frame, "ALERTA: FADIGA DETECTADA!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_DANGER, 3)
                    beep_alert()
            else:
                start_time_eyes = None
                eyes_closed_duration = 0

    # ==========================
    # PROCESSAMENTO DE MÃOS (VOLANTE)
    # ==========================
    results_hands = hands.process(frame_rgb)
    hands_on_wheel = False
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Verifica se ESTA mão está na zona segura
            if is_hand_in_zone(hand_landmarks, wheel_rect_norm):
                hands_on_wheel = True # Pelo menos uma mão está no volante
                
                # Desenha o esqueleto VERDE apenas se estiver na zona
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_SAFE, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=COLOR_SAFE, thickness=2, circle_radius=2)
                )
            else:
                # Mão fora do volante (pode desenhar vermelho ou não desenhar nada)
                # Opção: Desenhar vermelho para mostrar que está detectando mas está errado
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_DANGER, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=COLOR_DANGER, thickness=2, circle_radius=2)
                )

    # Lógica de Tempo Sem Mãos
    if not hands_on_wheel:
        if start_time_hands is None:
            start_time_hands = time.time()
        
        hands_missing_duration = time.time() - start_time_hands
        
        cv2.putText(frame, f"SEM MAOS: {hands_missing_duration:.1f}s", (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WARNING, 2)

        if hands_missing_duration > HANDS_TIME_LIMIT:
            cv2.putText(frame, "ALERTA: MAOS FORA DO VOLANTE!", (w//2 - 250, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WARNING, 3)
            beep_alert()
    else:
        start_time_hands = None
        hands_missing_duration = 0


    # Mostra o resultado final
    cv2.imshow('Sistema de Seguranca do Motorista', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()