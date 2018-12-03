
import argparse
import cv2

# MOBİLENET SSD EGITILMIS VERI BAGLAMA KODLARI ///////////////////////////////////////////////////////////////////////////////////////
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")

# MOBİLENET SSD EGITILMIS VERI BAGLAMA KODLARI
# MOBİLENET SSD EGITILMIS VERI BAGLAMA KODLARI
# MOBİLENET SSD EGITILMIS VERI BAGLAMA KODLARI//////////////////////////////////////////////////////////////////////////////////////
ObjeIsimleri = { 0: 'arkaplan',
    1: 'uçak', 2: 'bisiklet', 3: 'kuş', 4: 'bot',
    5: 'şişe', 6: 'otobüs', 7: 'araba', 8: 'kedi', 9: 'sandalye',
    10: 'inek', 11: 'sehpa', 12: 'köpek', 13: 'at',
    14: 'motor', 15: 'insan', 16: 'ekim',
    17: 'koyun', 18: 'koltuk', 19: 'tren', 20: 'tvmonitor' }


cap = cv2.VideoCapture(0)
args = parser.parse_args()
#Eğitilmiş modelimizi yüklüyoruz.
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    # saniye saniye kayıt altına alıyoruz
    ret, cerceve = cap.read()

    cerceve_tekrar = cv2.resize(cerceve,(300,300)) # resize cerceve for prediction

    # Kamaması için 300*300 bir çerçevede görüntüyü işleyip blob kanalında saklıyoruz
    blob = cv2.dnn.blobFromImage(cerceve_tekrar, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    # sinir ağının girişlerine videodaki blob kanalından giriş ekliyoruz.
    net.setInput(blob)
    # ağda bir sonraki veri tahmin ediliyor
    detections = net.forward()

    # cerceve tekrar  (300x300)
    kolons = cerceve_tekrar.shape[1]
    satir = cerceve_tekrar.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # sinir ağı için ağırlık ekleyerek güven saglıyoruz.
        if confidence > args.thr:  #tahmin dogrulugu tutan verileri if in içine alıyoruz
            class_id = int(detections[0, 0, i, 1])  # Class label
            # Object location
            xSolAsagi = int(detections[0, 0, i, 3] * kolons)
            ysolAsagi = int(detections[0, 0, i, 4] * satir)
            xsagYukari = int(detections[0, 0, i, 5] * kolons)
            ysagYukari = int(detections[0, 0, i, 6] * satir)
            # çerçevenin orjinal boyutunundki 1 pikselin uzunlgunu buluyruz.
            heightFactor = cerceve.shape[0] / 300.0
            widthFactor = cerceve.shape[1] / 300.0
            # ve çerçeveyi nsneenin boyutuna göre ayarlıyoruz
            xSolAsagi = int(widthFactor * xSolAsagi)
            ysolAsagi = int(heightFactor * ysolAsagi)
            xsagYukari = int(widthFactor * xsagYukari)
            ysagYukari = int(heightFactor * ysagYukari)
            # objenin etrafına erçeve çiziyoruz ve bunu detaction edien nesnenin x ve y sinden alıyoruz.
            cv2.rectangle(cerceve, (xSolAsagi, ysolAsagi), (xsagYukari, ysagYukari),
                          (0, 255, 0))
            # yeniden boyutlandırdıgımız çereve ekrana tekrar çiziliyor.
            if class_id in ObjeIsimleri:
                label = ObjeIsimleri[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ysolAsagi = max(ysolAsagi, labelSize[1])
                #cercevenin icini dolduruyoruz.
                cv2.rectangle(cerceve, (xSolAsagi, ysolAsagi - labelSize[1]),
                              (xSolAsagi + labelSize[0], ysolAsagi + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(cerceve, label, (xSolAsagi, ysolAsagi),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                print(label)  # cerceveledigimiz goruntunun anlık ne oldugunu konsola yazıyoruz
        cv2.namedWindow("cerceve", cv2.WINDOW_NORMAL)
        cv2.imshow("cerceve", cerceve)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break