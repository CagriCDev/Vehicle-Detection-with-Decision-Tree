
import cv2

cap = cv2.VideoCapture('kutahya1.mp4')
car_cascade = cv2.CascadeClassifier('egitilmisArabalarim.xml')
while(cap.isOpened()):

    ret, frame = cap.read()
    key = cv2.waitKey(1)
    key = cv2.waitKey(1)

    #Egitilmis verilerimii az yer kaplaak icin ve dogru sonuc icin ve
    # En az sayıda giris icin grayscale e cevirilip egitilmisir.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Karsılastırma GrayScale olarak yapılmıstır.
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    #Araba Etrafinda Cerceve Olusturmamız icin karsilasilan her araba modeline cerceve cizimisr.
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = 'Araba'
        #Burada ise Karsilasilan modelde cerccenin kosesine Araba yazilmisitr.
        cv2.putText(frame,label,(x,y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('video2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()