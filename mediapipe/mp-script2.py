import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

jawline = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 435, 361]

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    
    image = cv2.imread("test6.jpg")
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:  

        # Draw the lips
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))      

        # Draw the jawline
        for i in range(len(jawline) - 1):
            x1, y1 = face_landmarks.landmark[jawline[i]].x*w, face_landmarks.landmark[jawline[i]].y*h
            x2, y2 = face_landmarks.landmark[jawline[i+1]].x*w,face_landmarks.landmark[jawline[i+1]].y*h
            cv2.line(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

    cv2.imwrite("mp4" + '.png', annotated_image)





