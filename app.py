import streamlit as st
import joblib

# Başlık
st.title("📝 Student Summary Scorer")
st.markdown("Yazdığınız özeti girin, içeriği ve anlatımı otomatik puanlayalım!")

# Kullanıcıdan metin al
text_input = st.text_area("✍️ Özetinizi buraya yazın", height=250)

# Model ve TF-IDF yükle
model = joblib.load("ridge_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Tahmin butonu
if st.button("📊 Puanla"):
    if text_input.strip() == "":
        st.warning("Lütfen bir özet metni girin.")
    else:
        # Vektörleştir ve tahmin et
        X = tfidf.transform([text_input])
        preds = model.predict(X)[0]
        
        st.success("✅ Tahminler tamamlandı:")
        st.write(f"**İçerik (Content)**: {round(preds[0], 2)} / 5")
        st.write(f"**Anlatım (Wording)**: {round(preds[1], 2)} / 5")
