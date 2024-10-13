# AUTOENCODERS   - UNSUPERVİSED LEARNİNG Algoritması
# Autoencoder amacı bir feature(özellik)teki en önemli özelliğini alıyor
# mesela en önemli özelliği havlaması
# özetle en az şekilde bilgi kaydetme


# Autoencoder'lar, veriyi sıkıştırıp (encoding) ardından yeniden oluşturan (decoding) yapay sinir ağlarıdır.
# Encoder kısmı veriyi düşük boyutlu bir temsile dönüştürürken, Decoder bu temsilden orijinal veriyi
# geri kazanmaya çalışır. Genellikle denetimsiz öğrenme yöntemleri arasında yer alır. Kullanım alanları 
# arasında gürültü azaltma, boyut indirgeme, anomali tespiti ve veri üretimi bulunur.
# Autoencoder'lar, etiketlenmemiş verilerle çalışabilir ve veri sıkıştırma, özellik çıkarma gibi işlemlerde etkilidir.
# Autoencoder'lar, veri sıkıştırma ve özellik çıkarımı gibi görevlerde kullanılan bir tür yapay sinir ağıdır.

# Girdi → Encoder → Gizli Katman → Decoder → Çıktı

# Amaç, girdi ve çıktının mümkün olduğunca benzer olmasıdır.


# Girdi verisi encoder'a verilir, bu aşamada veriler boyutları azaltılarak gizli bir temsile dönüştürülür.
# Decoder bu gizli temsili alır ve orijinal girdi verisine yakın bir çıktı üretmeye çalışır.
# Eğitim sırasında, modelin çıktısı ile girdisi arasındaki fark, yani rekonstrüksiyon hatası minimize edilmeye çalışılır.

# Autoencoder'larda x−r=0 durumu, girdi verisi
# x'in model tarafından tamamen hatasız bir şekilde yeniden yapılandırıldığını ifade eder. Bu durumda:
# x: Autoencoder'a verilen orijinal giriş verisidir.
# r: Autoencoder tarafından yeniden yapılandırılan (reconstructed) veridir.


# Ancak, pratikte x−r=0, yani tamamen hatasız bir yeniden yapılandırma neredeyse hiç mümkün değildir.
# Çünkü autoencoder'lar genellikle verileri sıkıştırır ve bu sıkıştırma işlemi sırasında bilgi kaybı meydana gelir.
# Autoencoder'lar bir dengeleme mekanizması olarak çalışır: veriyi sıkıştırırken en önemli bilgiyi kaydetmeye 
# ve bu bilgiyi kullanarak veriyi yeniden yapılandırmaya çalışır. Bu süreçte küçük miktarda bilgi kaybı olması doğaldır.

# inputsize = hiddenlayer size = output yani nöron sayıları aynı olursa birşey öğrenmez

# inputsize > hidden layer size şeklinde olmalıdır 
# size (giriş boyutu) gizli katman boyutundan (hidden size) büyük olmalıdır. 
# Bu, modelin, verinin tüm özelliklerini ve çeşitliliğini daha iyi öğrenebilmesi için gereklidir.

# Eğer gizli katman(hidden layer) boyutu çok küçükse, model verinin karmaşıklığını yeterince öğrenemeyebilir
#  ve bu da overfitting (aşırı uyum) veya underfitting (yetersiz uyum) sorunlarına yol açabilir.



from keras.models import Model
from keras.layers import Input , Dense
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import json , codecs
import warnings  # uyarıları kapatıyoruz
warnings.filterwarnings("ignore")

(x_train,_),(x_test,_) =fashion_mnist.load_data()

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

x_train = x_train.reshape((len(x_train),x_train.shape[1:][0]*x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test),x_test.shape[1:][0]*x_test.shape[1:][1]))


plt.imshow(x_train[1500].reshape(28,28))
plt.axis('off')


#%%

input_img = Input(shape=(784,))

encoded = Dense(32,activation="relu")(input_img)
encoded = Dense(16,activation="relu")(encoded)


decoded = Dense(32,activation="relu")(encoded)
decoded = Dense(784,activation="sigmoid")(decoded)

autoencoder = Model(input_img,decoded)

# modelleri birbirine bağlamış yani birleştirmiş olduk
autoencoder.compile(optimizer="rmsprop",
                    loss="binary_crossentropy")

hist = autoencoder.fit(x_train,x_train,
                       epochs=200,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(x_train,x_train))


#%% save model

autoencoder.save_weights("deneme.h5")


#%% değerlendirme


print(hist.history.keys())

plt.plot(hist.history["loss"], label="Training Loss")
plt.plot(hist.history["val_loss"],label=" Validation Loss")
plt.legend()
plt.show()

#%% save history

import json,codecs

with open("deneme.json","w") as f:
    json.dump(hist.history,f)

#%% load history

with codecs.open("deneme.json","r",encoding="utf-8") as f:
    n = json.loads(f.read())

print(n.keys())
plt.plot(n["loss"], label="Training Loss")
plt.plot(n["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

#%%

encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_test)

plt.imshow(x_test[1500].reshape(28,28))
plt.axis('off')
plt.show()


plt.imshow(encoded_img[1500].reshape(4,4))
plt.axis('off')
plt.show()


decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    # orjinali göster
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.axis("off")
    
    # ekran yeniden yapılandırması
    ax=plt.subplot(2,n,i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.axis("off")
plt.show()


