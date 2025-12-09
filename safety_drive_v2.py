import cv2
import mediapipe as mp
import time
import math
import numpy as np
import threading

# --- Configurações de Som (Windows) ---
# Usamos threading para o beep não travar o vídeo
def play_alarm():
    try:
        import winsound
        # Frequência 2500Hz (agudo), 1000ms (1 segundo)
        winsound.Beep(2500, 1000)
    except ImportError:
        # Fallback para Linux/Mac (apenas print)
        print("BEEP! (winsound não disponível)")

def trigger_alarm_thread():
    # Inicia o som em paralelo para não congelar a imagem
    t = threading.Thread(target=play_alarm)
    t.daemon = True
    t.start()

# --- Constantes e Ajustes ---
# Dica: Se estiver muito sensível, diminua o EAR_THRESHOLD para 0.20
EAR_THRESHOLD = 0.22        # Olhos fechados se EAR < 0.22
EYE_CLOSED_LIMIT = 2.0      # Segundos até o alerta de sono
HANDS_OFF_LIMIT = 2.5       # Segundos até o alerta de mãos
WHEEL_ZONE_Y_PERC = 0.65    # O volante começa em 65% da altura da tela

# Cores (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
GRAY = (50, 50, 50)

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# Índices dos olhos na malha facial
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- Funções Auxiliares ---
def calculate_ear(landmarks, indices, w, h):
    """Calcula a razão de aspecto do olho (abertura)"""
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    
    # Distâncias verticais
    v1 = math.dist(coords[1], coords[5])
    v2 = math.dist(coords[2], coords[4])
    # Distância horizontal
    h_dist = math.dist(coords[0], coords[3])
    
    if h_dist == 0: return 0.0
    return (v1 + v2) / (2.0 * h_dist)

def draw_progress_bar(img, x, y, w, h, progress, color, label):
    """Desenha uma barra de progresso com texto"""
    # Fundo
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, cv2.FILLED)
    # Barra de preenchimento
    fill_w = int(w * min(progress, 1.0))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, cv2.FILLED)
    # Borda
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # Texto
    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def get_face_bbox(landmarks, w, h):
    """Retorna o retângulo (x, y, w, h) que envolve o rosto"""
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    # Usa contorno do rosto para definir a caixa
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        if cx < x_min: x_min = cx
        if cx > x_max: x_max = cx
        if cy < y_min: y_min = cy
        if cy > y_max: y_max = cy
        
    return x_min, y_min, x_max - x_min, y_max - y_min

# --- Loop Principal ---
cap = cv2.VideoCapture(0)

# Temporizadores
start_time_eyes = None
start_time_hands = None
eyes_duration = 0
hands_duration = 0
alarm_cooldown = 0  # Evita disparar som freneticamente

print("Sistema Iniciado. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Preparação
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Define Área do Volante
    wheel_y_pos = int(h * WHEEL_ZONE_Y_PERC)
    wheel_box = (int(w * 0.2), wheel_y_pos, int(w * 0.8), h) # x1, y1, x2, y2
    
    # Desenha zona do volante (Cinza transparente simulado ou apenas borda)
    cv2.rectangle(frame, (wheel_box[0], wheel_box[1]), (wheel_box[2], wheel_box[3]), (100, 100, 100), 2)
    cv2.putText(frame, "ZONA DO VOLANTE", (wheel_box[0] + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

    # Variáveis de estado do frame atual
    is_drowsy = False
    is_hands_off = True # Começa assumindo que está sem mãos
    face_detected = False
    
    # ===========================
    # PROCESSAMENTO ROSTO (SONO)
    # ===========================
    results_face = face_mesh.process(rgb_frame)
    
    if results_face.multi_face_landmarks:
        face_detected = True
        face_lms = results_face.multi_face_landmarks[0].landmark
        
        # EAR (Olhos)
        left_ear = calculate_ear(face_lms, LEFT_EYE, w, h)
        right_ear = calculate_ear(face_lms, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Bounding Box do Rosto
        fx, fy, fw, fh = get_face_bbox(face_lms, w, h)
        
        # Lógica de Tempo (Olhos)
        if avg_ear < EAR_THRESHOLD:
            if start_time_eyes is None:
                start_time_eyes = time.time()
            eyes_duration = time.time() - start_time_eyes
            
            # Estado: Olhos Fechados
            status_text = f"Olhos: FECHADOS ({eyes_duration:.1f}s)"
            box_color = RED if eyes_duration > 0.5 else YELLOW
            
            if eyes_duration > EYE_CLOSED_LIMIT:
                is_drowsy = True
        else:
            start_time_eyes = None
            eyes_duration = 0
            # Estado: Olhos Abertos
            status_text = "Olhos: ABERTOS"
            box_color = GREEN
            
        # Desenha HUD no Rosto
        # 1. O Quadrado
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
        # 2. O Texto de status acima do quadrado
        cv2.putText(frame, status_text, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        # 3. Valor do EAR (Debug visual útil)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (fx + fw + 5, fy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

    # ===========================
    # PROCESSAMENTO MÃOS
    # ===========================
    results_hands = hands.process(rgb_frame)
    
    if results_hands.multi_hand_landmarks:
        for hand_lms in results_hands.multi_hand_landmarks:
            # Verifica se o pulso (landmark 0) está na zona do volante
            wrist_y = int(hand_lms.landmark[0].y * h)
            wrist_x = int(hand_lms.landmark[0].x * w)
            
            # Checa colisão com a caixa do volante
            if (wheel_box[0] < wrist_x < wheel_box[2]) and (wheel_box[1] < wrist_y < wheel_box[3]):
                is_hands_off = False
                # Desenha esqueleto VERDE (OK)
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=2))
            else:
                # Mão detectada mas fora da zona (Vermelho)
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2))

    # Lógica de Tempo (Mãos)
    if is_hands_off:
        if start_time_hands is None:
            start_time_hands = time.time()
        hands_duration = time.time() - start_time_hands
    else:
        start_time_hands = None
        hands_duration = 0

    # ===========================
    # UI e ALERTAS GERAIS
    # ===========================
    
    # 1. Barras de Progresso (Topo da tela)
    # Barra de Sono
    draw_progress_bar(frame, 20, 40, 200, 20, eyes_duration / EYE_CLOSED_LIMIT, RED, "Nivel de Fadiga")
    # Barra de Mãos
    draw_progress_bar(frame, w - 220, 40, 200, 20, hands_duration / HANDS_OFF_LIMIT, YELLOW, "Tempo s/ Maos")

    # 2. Disparo de Alertas
    current_time = time.time()
    trigger_sound = False
    
    # Alerta Visual Gigante
    if is_drowsy:
        cv2.putText(frame, "ALERTA: FADIGA!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 4)
        trigger_sound = True
        
    if hands_duration > HANDS_OFF_LIMIT:
        cv2.putText(frame, "ALERTA: MAOS!", (w//2 - 130, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 4)
        trigger_sound = True

    # Controle do Som (Cooldown de 2 seg para não bugar o audio)
    if trigger_sound and (current_time - alarm_cooldown > 2.0):
        trigger_alarm_thread()
        alarm_cooldown = current_time

    cv2.imshow('Driver Safety System V2.0', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()