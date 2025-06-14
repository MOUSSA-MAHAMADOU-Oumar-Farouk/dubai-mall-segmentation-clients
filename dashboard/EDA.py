import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv("data/Mall_Customers.csv")

# Initialisation des états
toggle_apercu = 'toggle_apercu' not in st.session_state or st.session_state.toggle_apercu
st.session_state.toggle_apercu = toggle_apercu

toggle_stats = 'toggle_stats' not in st.session_state or st.session_state.toggle_stats
st.session_state.toggle_stats = toggle_stats

# Couleurs et icônes pour les box
box_colors = ['#bfcc94', '#edb458', '#548c2f', '#ffb100', '#ffebc6', '#ffd449']
icons = ['👥', '♂️', '♀️', '🕰️', '💰', '🛒']

# Titre de la page
st.title("Analyse Exploratoire des Données (EDA)")
def run():
    # Boutons toggle pour afficher/masquer
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Aperçu de la base", key="apercu_btn"):
            st.session_state.toggle_apercu = not st.session_state.toggle_apercu

    with col_btn2:
        if st.button("Statistiques descriptives", key="stats_btn"):
            st.session_state.toggle_stats = not st.session_state.toggle_stats

    # Aperçu de la base
    df_display = df.head() if st.session_state.toggle_apercu else None
    if df_display is not None:
        st.subheader("Aperçu de la base")
        st.dataframe(df_display)

    # Statistiques descriptives
    if st.session_state.toggle_stats:
        st.subheader("Indicateurs statistiques clés")
        total_clients = len(df)
        hommes = df[df['Gender'] == 'Male']
        femmes = df[df['Gender'] == 'Female']

        stats_data = [
            ("Total Clients", total_clients, icons[0], box_colors[0]),
            ("Hommes(%)", round(100 * len(hommes) / total_clients, 1), icons[1], box_colors[1]),
            ("Femmes(%)", round(100 * len(femmes) / total_clients, 1), icons[2], box_colors[2]),
            ("Âge Moyen", round(df["Age"].mean(), 1), icons[3], box_colors[3]),
            ("Revenu Moyen(k$)", round(df["Annual Income (k$)"].mean(), 1), icons[4], box_colors[4]),
            ("Score Moyen", round(df["Spending Score (1-100)"].mean(), 1), icons[5], box_colors[5])
        ]

        col_metrics = st.columns(3)
        for idx, (label, value, icon, color) in enumerate(stats_data):
            with col_metrics[idx % 3]:
                st.markdown(f"""
                <div style='background-color:{color}; margin:8px; padding: 15px; border-radius: 5px; text-align: center;'>
                    <h5>{icon}</h5>
                    <h6 style="font-weight: bold;">{label}</h6>
                    <h4 style="font-weight: bold;">{value}</h4>
                </div>
                """, unsafe_allow_html=True)

    # Onglets pour les analyses
    st.markdown("---")
    onglet = st.tabs(["Analyse univariée", "Analyse bivariée", "Analyse multivariée"])

    # Analyse univariée
    with onglet[0]:
        st.subheader("Analyse univariée")
        var_uni = st.selectbox("Choisir une variable", ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

        if var_uni == "Gender":
            gender_counts = df["Gender"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
                colors=["lightblue", "pink"], wedgeprops={'width':0.3}, pctdistance=0.8)
            ax.set_title("Répartition des clients par sexe")
            ax.axis('equal')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.histplot(df[var_uni], kde=True, ax=ax, color='steelblue')
            ax.set_title(f"Distribution de la variable {var_uni}")
            st.pyplot(fig)

    # Analyse bivariée
    with onglet[1]:
        st.subheader("Analyse bivariée")
        var1 = st.selectbox("Variable 1", df.columns, index=0)
        var2 = st.selectbox("Variable 2", df.columns, index=3)

        if df[var1].dtype == 'object' and df[var2].dtype != 'object':
            fig, ax = plt.subplots()
            sns.boxplot(x=var1, y=var2, data=df, ax=ax)
            ax.set_title(f"{var1} vs {var2}")
            st.pyplot(fig)
        elif df[var1].dtype != 'object' and df[var2].dtype != 'object':
            fig, ax = plt.subplots()
            sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
            ax.set_title(f"{var1} vs {var2}")
            st.pyplot(fig)

    # Analyse multivariée
    with onglet[2]:
        st.subheader("Analyse multivariée")

        st.markdown("### Matrice de corrélation")
        corr_matrix = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.markdown("### Pairplot")
        fig_pair = sns.pairplot(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
        st.pyplot(fig_pair)