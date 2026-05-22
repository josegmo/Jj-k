import cv2
import mediapipe as mp
import numpy as np
import time
import math

class GestosPro:
    def __init__(self):
        self.esperando_liberacion = False 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.selfie = self.mp_selfie.SelfieSegmentation(model_selection=1)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
        
        # --- CONFIGURACIÓN [Carga, Enfriamiento, Color BGR] ---
        self.configs = {
            "SUKUNA":  [1.0, 5.0, (0, 0, 180)],
            "PURPURA": [1.3, 20.0, (150, 0, 100)], 
            "VACIO":   [1.0, 8.0, (20, 10, 5)], 
            "MEGUMI":  [1.0, 6.0, (0, 0, 0)]
        }
        
        self.tecnica_actual = None
        self.inicio_gesto = 0
        self.bloqueado_hasta = 0
        self.color_bloqueo = None
        
        # --- VARIABLES DE ESTADO PARA PÚRPURA ---
        self.purpura_lanzado = False
        self.t_lanzamiento = 0
        self.ready_to_launch = False
        self.last_target = (0, 0)

    def calc_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_dr(self, hand_lms):
        return self.calc_dist(hand_lms.landmark[0], hand_lms.landmark[8])

    def animar_purpura(self, frame, t_actual, hand_lms=None):
        h, w, _ = frame.shape
        inicio_bloqueo = self.bloqueado_hasta - self.configs["PURPURA"][1]
        t_rel = t_actual - inicio_bloqueo
        centro_x, centro_y = w // 2, h // 2
        dist_ini = w // 4

        def draw_energy_ball(img, x, y, color, size=45, glow_mult=1.0):
            # CAPA 1: Aura Morada (Debajo)
            for i in range(3, 0, -1):
                overlay = img.copy()
                r = int((size + (i * 20)) * glow_mult) 
                cv2.circle(overlay, (x, y), r, color, -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            
            # CAPA 2: Núcleo Blanco (Encima)
            cv2.circle(img, (x, y), int(size * 0.4 * glow_mult), (255, 255, 255), -1)
            
            # Partículas de energía
            for _ in range(6):
                px = x + np.random.randint(int(-size * glow_mult), int(size * glow_mult))
                py = y + np.random.randint(int(-size * glow_mult), int(size * glow_mult))
                cv2.circle(img, (px, py), 2, color, -1)

        # --- FASE 1: APARICIÓN Y FUSIÓN ---
        if t_rel < 1.5: 
            pass
        elif t_rel < 3.5: 
            draw_energy_ball(frame, centro_x - dist_ini, centro_y, (255, 0, 0))
        elif t_rel < 5.5: 
            draw_energy_ball(frame, centro_x - dist_ini, centro_y, (255, 0, 0))
            draw_energy_ball(frame, centro_x + dist_ini, centro_y, (0, 0, 255))
        elif t_rel < 8.5: 
            pos_t = (t_rel - 5.5) / 3.0
            x_a = int((centro_x - dist_ini) + (dist_ini * pos_t))
            x_r = int((centro_x + dist_ini) - (dist_ini * pos_t))
            draw_energy_ball(frame, x_a, centro_y, (255, 0, 0))
            draw_energy_ball(frame, x_r, centro_y, (0, 0, 255))

        # --- FASE 2: SEGUIMIENTO (TRACKING) ---
        elif not self.purpura_lanzado:
            target_x, target_y = centro_x, centro_y
            if hand_lms:
                dist_puntas = self.calc_dist(hand_lms.landmark[8], hand_lms.landmark[12])
                dr = self.get_dr(hand_lms)
                if dist_puntas < (dr * 0.2):
                    target_x = int(((hand_lms.landmark[8].x + hand_lms.landmark[12].x) / 2) * w)
                    target_y = int(((hand_lms.landmark[8].y + hand_lms.landmark[12].y) / 2) * h)
                    self.last_target = (target_x, target_y)
                    self.ready_to_launch = True
                elif getattr(self, 'ready_to_launch', False):
                    self.purpura_lanzado = True
                    self.t_lanzamiento = t_actual 
                    self.ready_to_launch = False 
            draw_energy_ball(frame, target_x, target_y, (150, 0, 100), size=60)

        # --- FASE 3: EXPANSIÓN UNIFORME Y CIERRE ---
        else:
            t_exp = t_actual - self.t_lanzamiento
            lx, ly = getattr(self, 'last_target', (centro_x, centro_y))
            
            if t_exp < 1.5: 
                zoom = 1.0 + (t_exp ** 2 * 10) 
                draw_energy_ball(frame, lx, ly, (150, 0, 100), size=60, glow_mult=zoom)
            
            elif t_exp < 5.5: 
                overlay = frame.copy()
                overlay[:] = (150, 0, 100)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                for _ in range(40): 
                    cv2.circle(frame, (np.random.randint(0, w), np.random.randint(0, h)), 2, (255, 255, 255), -1)
            
            else:
                self.bloqueado_hasta = 0
                self.purpura_lanzado = False
                self.ready_to_launch = False
                self.esperando_liberacion = True 

        return frame

    def checar_tecnicas(self, lms_list):
        if len(lms_list) == 2:
            h1, h2 = lms_list[0], lms_list[1]
            dr = (self.get_dr(h1) + self.get_dr(h2)) / 2
            if self.calc_dist(h1.landmark[8], h2.landmark[8]) < (dr * 0.25) and \
               self.calc_dist(h1.landmark[12], h2.landmark[12]) < (dr * 0.25):
                return "SUKUNA"
            if self.calc_dist(h1.landmark[0], h2.landmark[0]) < (dr * 0.35) and \
               self.calc_dist(h1.landmark[4], h2.landmark[4]) < (dr * 0.35):
                return "MEGUMI"
            if self.calc_dist(h2.landmark[8], h2.landmark[5]) < (self.get_dr(h2) * 0.6):
                return "PURPURA"
        elif len(lms_list) == 1:
            h = lms_list[0]
            if self.calc_dist(h.landmark[8], h.landmark[12]) < (self.get_dr(h) * 0.2):
                return "VACIO"
        return None

def overlay_transparent(background, overlay, x, y, size):
    if overlay is None: return background
    img = cv2.resize(overlay, (size, size))
    h, w, _ = background.shape
    y1, y2 = max(0, y - size//2), min(h, y + size//2)
    x1, x2 = max(0, x - size//2), min(w, x + size//2)
    img_crop = img[0:(y2-y1), 0:(x2-x1)]
    if img_crop.shape[2] == 4:
        alpha = img_crop[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha * img_crop[:, :, c] + (1.0 - alpha) * background[y1:y2, x1:x2, c])
    return background

# --- SETUP INICIAL ---
detector = GestosPro()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# RECURSOS MULTIMEDIA
video_vacio = cv2.VideoCapture('dom_va.mp4')
video_sukuna = cv2.VideoCapture('sukuna_dom.mp4')
video_megumi = cv2.VideoCapture('megumi_efecto.mp4')
img_satoru = cv2.imread('cara_satoru.png', cv2.IMREAD_UNCHANGED)
img_cara_sukuna = cv2.imread('cara_sukuna.png', cv2.IMREAD_UNCHANGED)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    t_actual = time.time()
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_seg = detector.selfie.process(rgb)
    res_hands = detector.hands.process(rgb)
    res_face = detector.face_mesh.process(rgb)
    
    cond = (res_seg.segmentation_mask > 0.5)[:, :, None] if res_seg.segmentation_mask is Jacks else None
    if res_seg.segmentation_mask is not None:
        cond = (res_seg.segmentation_mask > 0.5)[:, :, None]
    else:
        cond = np.ones((h, w, 1), dtype=bool)

    msg = "Esperando tecnica..."

    # --- JERARQUÍA DE ESTADOS ---
    if t_actual < detector.bloqueado_hasta:
        if detector.tecnica_actual == "VACIO":
            ret, f_v = video_vacio.read()
            if not ret: 
                video_vacio.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f_v = video_vacio.read()
            if ret:
                capa_negra = np.zeros_like(frame, dtype=np.uint8)
                f_v = cv2.resize(f_v, (w, h))
                silueta = cv2.addWeighted(frame, 0.3, capa_negra, 0.7, 0)
                frame = np.where(cond, silueta, f_v)
            if res_face.multi_face_landmarks:
                f_l = res_face.multi_face_landmarks[0]
                frame = overlay_transparent(frame, img_satoru, int(f_l.landmark[1].x*w), int(f_l.landmark[1].y*h), int(abs(f_l.landmark[10].y-f_l.landmark[152].y)*h*1.6))

        elif detector.tecnica_actual == "SUKUNA":
            ret, f_s = video_sukuna.read()
            if not ret: 
                video_sukuna.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f_s = video_sukuna.read()
            if ret:
                f_s = cv2.resize(f_s, (w, h))
                silueta = cv2.addWeighted(frame, 0.7, np.full_like(frame, (0,0,180), np.uint8), 0.3, 0)
                frame = np.where(cond, silueta, f_s)
            if res_face.multi_face_landmarks:
                f_l = res_face.multi_face_landmarks[0]
                frame = overlay_transparent(frame, img_cara_sukuna, int(f_l.landmark[1].x*w), int(f_l.landmark[1].y*h), int(abs(f_l.landmark[10].y-f_l.landmark[152].y)*h*1.6))

        elif detector.tecnica_actual == "MEGUMI":
            ret, f_m = video_megumi.read()
            if not ret: 
                video_megumi.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f_m = video_megumi.read()
            if ret: 
                frame = cv2.addWeighted(frame, 0.5, cv2.resize(f_m, (w, h)), 0.5, 0)

        elif detector.tecnica_actual == "PURPURA":
            m_l = res_hands.multi_hand_landmarks[0] if res_hands.multi_hand_landmarks else None
            frame = detector.animar_purpura(frame, t_actual, m_l)
        
        msg = f"TECNICA ACTIVA: {detector.tecnica_actual}"
    
    else:
        # RESET DE ESTADOS AL TERMINAR EL BLOQUEO
        detector.purpura_lanzado = False
        detector.ready_to_launch = False
        
        # Gestor de liberación por seguridad
        if detector.esperando_liberacion:
            if not res_hands.multi_hand_landmarks:
                detector.esperando_liberacion = False
        
        # SOLO DETECTA NUEVAS TÉCNICAS SI NO HAY BLOQUEO ACTIVO Y SE LIBERÓ EL GESTO ANTERIOR
        if not detector.esperando_liberacion:
            gesto = detector.checar_tecnicas(res_hands.multi_hand_landmarks) if res_hands.multi_hand_landmarks else None
            if gesto:
                t_c, t_e, col = detector.configs[gesto]
                if detector.tecnica_actual == gesto:
                    prog = t_actual - detector.inicio_gesto
                    if prog >= t_c:
                        detector.bloqueado_hasta = t_actual + t_e
                        detector.color_bloqueo = col
                    msg = f"CARGANDO {gesto}: {round(prog, 1)}s"
                else:
                    detector.tecnica_actual = gesto
                    detector.inicio_gesto = t_actual
            else: 
                detector.tecnica_actual = None

    # DIBUJAR PUNTOS CLAVE (CAPA FINAL)
    if res_hands.multi_hand_landmarks:
        for l in res_hands.multi_hand_landmarks: 
            mp.solutions.drawing_utils.draw_landmarks(frame, l, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.putText(frame, msg, (20, 50), 0, 0.8, (255, 255, 255), 2)
    cv2.imshow('Jujutsu Engine - TintorEngin', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
