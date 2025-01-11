# import library yang dibutuhkan
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


@st.cache_data
# Fungsi untuk memuat data
def load_data():
    data = pd.read_csv("final.csv")
    return data


# Inisialisasi data
def initialize_data():
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    df["final_text"] = df["final_text"].astype(str)


# Membuat bar chart
def create_bar_chart_sentiment():
    distribution_df = (
        df.groupby("sentiment", observed=False)
        .agg({"sentiment": "count"})
        .rename(columns={"sentiment": "count"})
        .reset_index()
    )

    fig = px.bar(
        distribution_df,
        x="sentiment",
        y="count",
        labels={"sentiment": "Sentiment", "count": "Count"},
        title="Distribusi Sentimen",
        color="sentiment",
        color_discrete_map={
            "positif": "blue",
            "netral": "gray",
            "negatif": "red",
        },
    )

    st.plotly_chart(fig)


# Membuat line chart
def create_line_chart_sentiment():
    distribution_df = (
        df.groupby(["date", "sentiment"], observed=False)
        .agg({"sentiment": "count"})
        .rename(columns={"sentiment": "count"})
        .reset_index()
    )

    fig = px.line(
        distribution_df,
        x="date",
        y="count",
        labels={"date": "Date", "count": "Count"},
        title="Distribusi Sentimen per Hari",
        markers=True,
        color="sentiment",
        color_discrete_map={
            "positif": "blue",
            "netral": "gray",
            "negatif": "red",
        },
    )

    st.plotly_chart(fig)


# Membuat word cloud
def create_word_cloud_sentiment():
    for sentiment in df["sentiment"].unique():
        texts = " ".join(df[df["sentiment"] == sentiment]["final_text"])
        wordcloud = WordCloud(width=800, height=300).generate(texts)

        fig = px.imshow(
            wordcloud, title=f"Word Cloud Sentimen {sentiment.capitalize()}"
        )

        st.plotly_chart(fig)


# Membuat fitur TF-IDF
def create_tfidf_features():
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, ngram_range=(1, 2))
    # Fit dan transform data menggunakan TF-IDF
    tfidf_features = tfidf_vectorizer.fit_transform(df["final_text"])
    # Konversi ke DataFrame
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)

    return tfidf_df, tfidf_vectorizer, tfidf_features


# Menganalisis kata-kata penting berdasarkan nilai TF-IDF
def analyze_important_words(tfidf_df, n_words=10):
    # Menghitung rata-rata nilai TF-IDF untuk setiap kata
    mean_tfidf = tfidf_df.mean()
    # Mendapatkan n kata dengan nilai TF-IDF tertinggi
    top_words = mean_tfidf.nlargest(n_words)

    return top_words


# Membuat bar chart kata-kata penting
def create_bar_chart_important_words():
    tfidf_df, _, _ = create_tfidf_features()
    important_words = analyze_important_words(tfidf_df)

    fig = px.bar(
        important_words,
        x=important_words.index,
        y=important_words.values,
        labels={"index": "Kata", "y": "Nilai TF-IDF"},
        title="Daftar 10 Kata Penting Berdasarkan TF-IDF",
    )

    st.plotly_chart(fig)


# Membuat bar chart kata-kata penting berdasarkan sentimen
def create_bar_chart_important_words_by_sentiment(cols):
    tfidf_df, _, _ = create_tfidf_features()

    for index, col in enumerate(cols):
        with col:
            sentiment = df["sentiment"].unique()[index]
            sentiment_tfidf = tfidf_df[df["sentiment"] == sentiment]

            mean_tfidf = sentiment_tfidf.mean()
            top_words = mean_tfidf.nlargest(5)

            fig = px.bar(
                top_words,
                x=top_words.index,
                y=top_words.values,
                labels={"index": "", "y": ""},
                title=f"Sentimen {sentiment.capitalize()}",
            )

            fig.update_xaxes(tickangle=45)

            st.plotly_chart(fig)


# Train model
def train_naive_bayes_model():
    _, _, tfidf_features = create_tfidf_features()

    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_features,
        df["sentiment"],
        test_size=0.3,
        random_state=42,
        stratify=df["sentiment"],
    )

    # Membangun Model Naive Bayes
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train, y_train)

    # Prediksi dan Evaluasi
    y_pred = naive_bayes_model.predict(X_test)

    # Menampilkan bias dan akurasi
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=["positif", "negatif", "netral"],
        y=["positif", "negatif", "netral"],
        title="Confusion Matrix Naive Bayes",
    )

    st.plotly_chart(fig)

    # Menampilkan distribusi probabilitas prediksi
    pred_proba = naive_bayes_model.predict_proba(X_test)
    pred_proba_df = pd.DataFrame(pred_proba, columns=naive_bayes_model.classes_)

    hist_data = [pred_proba_df[sentiment] for sentiment in naive_bayes_model.classes_]
    colors = ["red", "gray", "blue"]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(
        hist_data,
        naive_bayes_model.classes_,
        show_hist=False,
        show_rug=False,
        colors=colors,
    )

    # Add title
    fig.update_layout(title_text="Distribusi Probabilitas Prediksi Naive Bayes")

    st.plotly_chart(fig)


# st.plotly_chart(fig)


# Ambil dataset
df = load_data()

# Inisialisasi data
initialize_data()

# Menampilkan judul dan deskripsi
st.title("Dashboard Analisis Sentimen")
st.write("Dashboard ini menampilkan analisis sentimen dari dataset yang telah diolah.")

# Menampilkan data
create_bar_chart_sentiment()
create_line_chart_sentiment()
create_word_cloud_sentiment()
create_bar_chart_important_words()

cols = st.columns(3)
create_bar_chart_important_words_by_sentiment(cols)

# Train model
train_naive_bayes_model()
