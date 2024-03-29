
from tkinter import *
from tkinter import messagebox, Canvas
from PIL import Image, ImageTk
from sqlalchemy import false
import tensorflow.compat.v1 as tf
import tensorflow.keras.models as tf_keras
import mediapipe as mp
import numpy as np
import detect_and_align as detectar_y_alinear
import cv2
import pymysql
from DatosPersona import IdPersona, cargar_modelo


#from mlflow import log_metric, log_param, log_artifact
#from mlflow.tracking import MlflowClient
import mlflow
import psutil
import os

l1, l2, l3 = psutil.getloadavg()
# Documentation for a class.
#
#  More details.


class LCCRecognition(Frame):

    def __init__(self, root=None, model=None, id_folder=None, umbral=None):
        """## Constructor por defecto de la clase.
            @Parametro root Nombre de ventana de tkinter. Se debe de crear antes de hacer una instancia.
            @Parametro model Nombre del modelo para cargar. Se debe de mandar el nombre, no la ruta.
            @Parametro id_folder Ruta donde están localizado los nombres de las personas que serán reconocidas.
            @Parametro umbral Parámetro para ajustar el nivel de detección de caras. Si da falsos positivos, es recomendable subirlo un poco
            """
        super().__init__(root)
        self.id_folder = id_folder
        self.umbral = umbral
        self.model = model
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.AlumnoEnCamara = False
        self.TiempoParaBorrarDato = 0
        self.SePuedeConsultar = True
        self.YaSeConsulto = False
        
        self.matricula = "Desconocido"
        self.msgMano = ""
        
        
        self.CargarModeloReconocimientoFacial(id_folder)
        self.CargarModeloReconocimientoGestosManos()

        # Variables para datos del alumno identificado
        self.creditos_totales = 383
        global cap
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #self.Visualizar()

    def CargarModeloReconocimientoFacial(self, id_folder):
        """### Método de clase para cargar un modelo para el reconocimiento facial .
        Se abre una sesión en tensorflow para asignar valores que serán procesados en su momento
        @Parametro id_folder Carpeta donde se localizarán fotos de personas con su etiqueta.
         """
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.mtcnn = detectar_y_alinear.create_mtcnn(self.sess, None)
            cargar_modelo(self.model)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            self.id_data = IdPersona(id_folder[0],
                                     self.mtcnn,
                                     self.sess,
                                     self.embeddings,
                                     self.images_placeholder,
                                     self.phase_train_placeholder,
                                     self.umbral)

    def CargarModeloReconocimientoGestosManos(self):
        """### Método de clase para cargar un modelo para el reconocimiento gesticular de las manos .
       Se cargará todo lo necesario para el reconocimiento de gestos de las manos

       ##Falta agregar más documentación
        """
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.modelo_manos = tf_keras.load_model('mp_hand_gesture')
        self.f = open('gesture.names', 'r')
        self.classNames = self.f.read().split('\n')
        self.f.close()

    def Iniciar(self):
        """### Método de clase para inciar el proceso de reconocimiento facial
            NOTA: LA CÁMARA SE VA A PRENDER.

            ##Falta agregar más documentación
        """

        self.Visualizar()

    def Visualizar(self):
        while True:

            if cap is not None:
                self.NoHayPersona()
                ret, frame = self.LeerFrameCamara()
                if self.FrameDisponible(ret):
                    face_patches, cuadros_delimitadores, puntos_referencia = detectar_y_alinear.detect_faces(
                        frame, self.mtcnn)
                    if len(face_patches) > 0:
                        # Codigo por comentar....
                        """
                        Las embeddings de palabras son, de hecho, una clase de técnicas en las que las palabras individuales
                        se representan como vectores de valor real en un espacio vectorial predefinido. 
                        Cada palabra se mapea a un vector y los valores vectoriales se aprenden de una manera que se asemejan a los de un vector.

                        An embedding is a low-dimensional translation of a high-dimensional vector

                        Recibe datos que no son de bajo nivel, más complejos y se crea un embedding vector que te describe lo que estamos obteniendo
                        Y la red neuronal se alimenta de esto. (Face Features)
                        Face Embedding, presentar caracteristicas del mundo real, como la cara, la mano, el rostro, etc. representarlo en algún nivel de detalle.
                        """

                        # FEATURE EXTRACTION DEL FRAME
                        embs = self.Embeddings(face_patches)
                        # print(embs)
                        personas_reconocidas, distancias_personas_reconocidas = self.id_data.find_matching_ids(
                            embs)

                        """
                        posicion_cara: Todas las caras reconocidas en el frame, será un arreglo de indice 4, donde estará su posición (x,y) y su tamaño
                        persona_reconocida: Es una etiqueta con el nombre de la persona reconocida
                        
                        cuadros_delimitadores se llama comunmente como bonding box
                        """
                        for posicion_cara, _, persona_reconocida, distancia in zip(
                            cuadros_delimitadores, puntos_referencia, personas_reconocidas, distancias_personas_reconocidas
                        ):
                            # Comenzamos suponiendo que no hay ninguna persona
                            self.NoHayPersona()
                            if self.PersonaCercaParaReconocer(posicion_cara):
                                if persona_reconocida is not None:
                                    self.HayPersona()
                                    if self.PersonaEnPosicion():
                                        gestoMano = self.DetectarMano(frame)
                                        self.msgMano = gestoMano    
                                        self.matricula = persona_reconocida
                                        if not self.YaSeConsulto:
                                            self.YaSeConsulto = True
                                        # Se mostrará información de lo que hará el programa, ejemplo "Buscar información, feedback negativo, postiivo,"
                                        self.MensajeMano(
                                            frame, gestoMano, persona_reconocida, self.font, distancia)
                                        
                                        #client.log_metric(run_id=run.info.run_id, key="Distancia  -" + str(persona_reconocida) + "-", value=distancia)
                                        #log_metric(key="Distancia  -" + str(persona_reconocida) + "-", value=distancia)

                                else:
                                    self.NoHayPersona()
                                    self.matricula = "Desconocido"
                                # Una vez reconocido el rostro, se va a mostrar en la cámara: Su nombre, Un cuadrado identificado al rostro y un pequeño mensaje
                                self.EncuadrarPersonaReconocida(
                                    frame, posicion_cara)
                                self.EtiquetarPersonaReconocida(
                                    frame, persona_reconocida, posicion_cara, self.font)
                                #print("Hola %s! Acuraccy: %1.4f" % (persona_reconocida, distancia))
                            '''
                            else:
                                self.Advertencia(
                                    frame, "Acercate a la camara", posicion_cara, self.font)
                                self.NoHayPersona()
                                self.matricula = "Desconocido"
                             '''

                    # Verificamos gestos de la mano
                    self.PersonaSeFue()  # verificar si la persona se fue
                    # self.MostrarFrame(frame)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                   
                    yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


                else:
                    self.BorrarVideo()

    def Embeddings(self, face_patches):
        face_patches = np.stack(face_patches)
        feed_dict = {self.images_placeholder: face_patches,
                     self.phase_train_placeholder: False}
        embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embs

    def FrameDisponible(self, ret):
        return ret

    def PersonaCercaParaReconocer(self, posicion_cara):
        #print(posicion_cara)
        if posicion_cara[2] > 400 and posicion_cara[3] > 400:
            return True
        return False

    def PersonaEnPosicion(self):
        return self.AlumnoEnCamara

    def EncuadrarPersonaReconocida(self, frame, posicion_cara):
        cv2.rectangle(frame, (posicion_cara[0], posicion_cara[1]),
                      (posicion_cara[2], posicion_cara[3]), (0, 255, 0), 4)

    def EtiquetarPersonaReconocida(self, frame, persona_reconocida, posicion_cara, font):
        cv2.putText(frame, persona_reconocida,
                    (posicion_cara[0] + 15, posicion_cara[1]-5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def BorrarVideo(self):
        self.lblVideo.image = ""
        cap.release()

    def LeerFrameCamara(self):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        return ret, frame

    def HayPersona(self):
        self.AlumnoEnCamara = True
        self.TiempoParaBorrarDato = 0

    def NoHayPersona(self):
        self.AlumnoEnCamara = False
        self.TiempoParaBorrarDato += 1
        self.SePuedeConsultar = False
        self.YaSeConsulto = True
        #self.matricula = "Desconocido"


    def Advertencia(self, frame, msg, posicion_cara, font):
        cv2.putText(frame, msg, (150, 450), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (posicion_cara[0], posicion_cara[1]),
                      (posicion_cara[2], posicion_cara[3]), (0, 0, 255), 4)

    def PersonaSeFue(self):
        
        if not self.AlumnoEnCamara:
            self.TiempoParaBorrarDato += 1
            self.YaSeConsulto = False
            #self.matricula = "Desconocido"
            '''
    YA NO SE VA A OCUPAR ESTA FUNCION DEBIDO A QUE SE BORRAN LOSD DATOS DESDE INDEX.HTML
        if self.TiempoParaBorrarDato > 30:
            #self.borrarDatos()
            self.YaSeConsulto = False
            print("Datos borrados")
            self.TiempoParaBorrarDato = 0
            '''

    def DetectarMano(self, frame):
        x, y, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(framergb)
        className = ''
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                self.mpDraw.draw_landmarks(
                    frame, handslms, self.mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = self.modelo_manos.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = self.classNames[classID]
        return className

    def MensajeMano(self, frame, className, persona_reconocida, font, distancia):
        msg = ""
        feedback = 0
        if (className == 'thumbs up'):
            msg = "Feedback positivo!"

        if (className == 'live long'):
            if (self.SePuedeConsultar):

                msg = "Buscando tu informacion..."
                self.BuscarDatosAlumno(persona_reconocida)
                self.TiempoParaBorrarDato = 0
                self.SePuedeConsultar = False

        if (className == 'thumbs down'):
            msg = "Feedback negativo!"

    def MostrarFrame(self, frame):
        cv2.imshow('Imagen', frame)
        self.Visualizar()

    def BuscarDatosAlumno(self, persona_reconocida):
        '''
        db = pymysql.connect(host = 'localhost', 
                         user= 'root',
                         password='xSK!NyF@pU#sD&L', 
                         database='alumnos_lcc',connect_timeout=10000)
        cur = db.cursor()
        sql = 'SELECT * FROM alumnos_lcc.alumno LEFT JOIN alumnos_lcc.registros ON alumnos_lcc.alumno.idalumno =alumnos_lcc.registros.id_alumno WHERE idalumno = {} ORDER BY fecha_ingreso DESC limit 1 ;'.format(persona_reconocida)
        try:
            cur.execute(sql)
            results = cur.fetchall()
            if not results:
                messagebox.showinfo(message="Tus datos no están capturados correctamente. Contactate con el administrador", title="¡Error!")
                return
            self.RellenarFormularioAlumno(results, persona_reconocida)
            self.Saludar()
            self.RegistrarAsistencia(cur, persona_reconocida,db)
        except pymysql.Error as e:
            print(e)

        '''
        print("hola")

    def RellenarFormularioAlumno(self, results, persona_reconocida):
        self.lcc_nombre_identificado.set(results[0][1])
        self.lcc_apellido_identificado.set(results[0][2])
        self.lcc_creditos_identificado.set(str(results[0][3]) + '/{} - {}%'.format(
            self.creditos_totales, round(results[0][3]/self.creditos_totales*100, 2)))
        self.lcc_kardex_identificado.set(results[0][4])
        self.lcc_fecha_ultimoingreso.set(results[0][7])
        if results[0][3] < int(self.creditos_totales * .8):
            self.lcc_sc_identificado.set("No aplica ")
        if results[0][3] >= int(self.creditos_totales * .8):
            self.lcc_sc_identificado.set("Si aplica")
        if results[0][3] < int(self.creditos_totales * .9):
            self.lcc_pp_identificado.set("No aplica ")
        if results[0][3] >= int(self.creditos_totales * .9):
            self.lcc_pp_identificado.set("Si aplica")

    def RegistrarAsistencia(self, cur, persona_reconocida, db):
        sql2 = 'INSERT INTO alumnos_lcc.registros(id_alumno) VALUES (%s);'
        cur.execute(sql2, (persona_reconocida))
        db.commit()

    def Saludar(self):
        self.saludocompleto.set(
            "Hola {} {}\nLa ultima vez que te vi fue el\n{}".format(
                self.lcc_nombre_identificado.get(),
                self.lcc_apellido_identificado.get(),
                self.lcc_fecha_ultimoingreso.get())
        )


if __name__ == "__main__":
    cap = None

    LCCRecognition(model='./model/20170512-110547.pb',
                   id_folder=['./ids/'], umbral=1.09)
