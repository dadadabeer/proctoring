import face_recognition
import cv2
import os



def face_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        return "0%"  # No match, confidence is 0%
    else:
        # Adjusted formula for confidence calculation
        linear_val = (1.0 - face_distance) / (1.0 - face_match_threshold)
        return str(round(linear_val * 100, 2)) + "%"


class f_recognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encode_faces()  # Encode faces at the beginning

    def encode_faces(self):
        for image_file in os.listdir('faces'):
            # Assuming 'faces' directory contains images named after the person in the image
            img_path = os.path.join('faces', image_file)
            face_image = face_recognition.load_image_file(img_path)

            # Calculate face encodings for the image
            encodings = face_recognition.face_encodings(face_image)

            # Check if any face encodings were found
            if encodings:
                face_encoding = encodings[0]  # Use the first encoding

                # Use the file name (without extension) as the known face name
                name = os.path.splitext(image_file)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
            else:
                print(f"No faces found in {image_file}.")
                continue  # Skip the rest of the loop and proceed with the next image

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Process every frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            print("rgb_small_frame shape:", rgb_small_frame.shape)
            print("face_locations:", face_locations)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = "0%"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    face_distance = \
                    face_recognition.face_distance([self.known_face_encodings[first_match_index]], face_encoding)[0]
                    confidence = face_confidence(face_distance)

                face_names.append(f'{name} ({confidence})')

            # Annotate and display the video frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = f_recognition()
    fr.run_recognition()
