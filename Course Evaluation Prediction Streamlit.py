import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scipy.stats as sps
from scipy.stats import loguniform
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import NMF, LatentDirichletAllocation
import xgboost as xg
import time
from itertools import chain
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import wordnet
import gensim

def main():
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    st.title("Course Evaluation for Milestone 2")
    st.markdown("Can we predict what score a course will receive?")

    st.sidebar.title("EDA & Supervised Learning")
    st.sidebar.markdown("Exploratory Data Analysis & Supervised Learning Approaches to Predict Score")

    RANDOM_SEED = 42
    stop_wrds = stopwords.words("english")
    lem = wordnet.WordNetLemmatizer()

    @st.cache(persist=True, suppress_st_warning=True)
    def normalize_text(txt):
        """
        Inputs: a single course evaluation from a student
        Outputs: a stripped, lowercased, no-stopwords, lemmatized version of that evaluation
        """
        #stripping and lowering extracted words
        txt_clean_first = re.sub(r'\[(.*?)\]', '', str(txt))
        txt_clean = re.sub(r'[^a-zA-Z\s]', '', str(txt_clean_first).lower().strip())
        txt_clean_split = txt_clean.split()
        #taking out stop words from words list
        txt_clean_split_nostop = [word for word in txt_clean_split if word not in stop_wrds]
        #lemmatizing words from list
        txt_clean_split_nostop_lem = [lem.lemmatize(word) for word in txt_clean_split_nostop]
        #joining back into sentence
        text = " ".join(txt_clean_split_nostop_lem)

        return text
    

    @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
    def load_data():
        """
        Inputs: None
        Outputs: three dataframes; train/val, test, and the evaluation with deomgraphics

        This function normalizes the unstructured text, creates some numerical features from the text, and removes outliers
        """
        status_text = st.empty()

        eval_demog_df = pd.read_csv('course_evaluation_with_demographics.csv')
        train_val_df = pd.read_csv('course_evaluation_without_demographics_TRAINING.csv')
        test_df = pd.read_csv('course_evaluation_without_demographics_TEST.csv')

        #dropping Nans
        train_val_df.dropna(inplace=True)

        #Lemmatizing and Cleaning free text
        train_val_df["text_clean"] = train_val_df["Comment.Text.Processed"].apply(lambda x: normalize_text(x))
        test_df["text_clean"] = test_df["Comment.Text.Processed"].apply(lambda x: normalize_text(x))
        eval_demog_df["text_clean"] = eval_demog_df["Comment.Text.Processed"].apply(lambda x: normalize_text(x))

        #Creating numerical features
        train_val_df['word_count'] = train_val_df["Comment.Text.Processed"].apply(lambda x: len(str(x).split(" ")))
        train_val_df['char_count'] = train_val_df["Comment.Text.Processed"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
        train_val_df['sentence_count'] = train_val_df["Comment.Text.Processed"].apply(lambda x: len(str(x).split(".")))
        train_val_df['avg_word_length'] = train_val_df['char_count'] / train_val_df['word_count']
        train_val_df['avg_sentence_length'] = train_val_df['word_count'] / train_val_df['sentence_count']
        train_val_df['capitals'] = train_val_df['Comment.Text.Processed'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        train_val_df['caps_vs_length'] = train_val_df.apply(lambda row: float(row['capitals'])/float(row['sentence_count']),axis=1)
        # train_val_df['num_unique_words'] = train_val_df['Comment.Text.Processed'].apply(lambda x: len(set(w for w in x.split())))
        # train_val_df["word_unique_percent"] =  train_val_df["num_unique_words"]*100/train_val_df['word_count']

        test_df['word_count'] = test_df["Comment.Text.Processed"].apply(lambda x: len(str(x).split(" ")))
        test_df['char_count'] = test_df["Comment.Text.Processed"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
        test_df['sentence_count'] = test_df["Comment.Text.Processed"].apply(lambda x: len(str(x).split(".")))
        test_df['avg_word_length'] = test_df['char_count'] / test_df['word_count']
        test_df['avg_sentence_length'] = test_df['word_count'] / test_df['sentence_count']
        test_df['capitals'] = test_df['Comment.Text.Processed'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        test_df['caps_vs_length'] = test_df.apply(lambda row: float(row['capitals'])/float(row['sentence_count']),axis=1)
        # test_df['num_unique_words'] = test_df['Comment.Text.Processed'].apply(lambda x: len(set(w for w in x.split())))
        # test_df["word_unique_percent"] =  test_df["num_unique_words"]*100/test_df['word_count']

        #Dropping columns with high correlations seen in EDA
        train_val_df = train_val_df.drop(columns=['word_count', 'sentence_count', 'capitals'])
        test_df = test_df.drop(columns=['word_count', 'sentence_count', 'capitals'])

        #removing outliers and fixing 0s
        train_val_df = train_val_df[~train_val_df['char_count'] < 10]
        train_val_df = train_val_df[~train_val_df.index.isin(train_val_df[train_val_df['text_clean'] == ''].index)]
        
        train_val_df = train_val_df[~(train_val_df['score'] == 0)]

        status_text.text('Loading data and feature engineering is done.')
        return train_val_df, test_df, eval_demog_df
    
    @st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
    def train_model(model, X_train, X_test, y_train, y_test, test):
        """
        Inputs: the raw versions after the splitting into X_train, X_test, y_train, y_test, and also the test dataframe
        Outputs: fitted model, predictions on validation set, and test predictions

        This function Scales and PowerTransforms the numerical attributes and the target variable, OneHotEncodes the categorical variables, and TfidfVectorizer for the normalized text.
        """
        cat_features = ['Division', 'Type']
        num_features = ['char_count', 'avg_word_length', 'avg_sentence_length', 'caps_vs_length']
        txt_features = 'text_clean'

        test_df = test.drop(columns=['id'])
        n_pipe = Pipeline([("robust", RobustScaler()),
                        ("power", PowerTransformer(standardize=True))])


        pipe = ([("num", n_pipe, num_features),
            ("cat", OneHotEncoder(categories='auto', handle_unknown='ignore'), cat_features),
            ("txt", TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=25000, 
                                    use_idf=False, norm='l2'), txt_features)])
        col_transform = ColumnTransformer(transformers=pipe)
        
        X_train = col_transform.fit_transform(X_train).toarray()
        X_test = col_transform.transform(X_test).toarray()
        test = col_transform.transform(test_df).toarray()    

        reg_model = TransformedTargetRegressor(regressor=model, transformer=PowerTransformer(standardize=True))
            
        reg_model = reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        
        test_final = reg_model.predict(test)
        
        return reg_model, y_pred, test_final, col_transform
    
    def fix_nans(yp_array, yp_idx):
        """
        Inputs: prediction array and index of the value resulting in nan
        Outputs: the real value of the inverse transform output

        This function is for handling the nans that result occaisonally when the inverse transform occurs at prediction time
        """

        ypredval = float(yp_array[yp_idx])
        x = (ypredval * 4.10518464 + 1) ** (1 / 4.10518464) - 1
        return x.real

    def plot_metrics(x, y, df=None, dummy=False, ytest=None, ypred=None):
        """
        Inputs: y_test, y_pred

        Visualizes y_pred against y_pred and also is used to visualize dummy predictors (mean strategy)
        """
        st.subheader("Actual vs Predicted Course Evaluation Scores")
        if dummy:
            fig = px.scatter(x=ytest, y=ypred,
                            trendline='ols', labels=dict(x="Actual (red)", y='Dummy Predicted (teal)'), color_discrete_sequence=['teal'])
            fig.add_shape(type='line',
                    x0=ytest.min(),
                    y0=ytest.min(),
                    x1=ytest.max(),
                    y1=ytest.max(),
                    line=dict(color='Red',),
                    xref='x',
                    yref='y')
        else:
            df['e'] = df[y]/100
            fig = px.scatter(df, x=x, y=y,
                trendline='ols', error_y="e", labels=dict(x="Actual (red)", y='Predicted (teal)'), color_discrete_sequence=['teal'])

            fig.add_shape(type='line',
                    x0=df[x].min(),
                    y0=df[x].min(),
                    x1=df[x].max(),
                    y1=df[x].max(),
                    line=dict(color='Red',),
                    xref='x',
                    yref='y')
        fig.update_layout(font=dict(size=12), width=800, height=800)
        st.plotly_chart(fig)

    def plot_residuals(df, x, y):
        """
        Inputs: dataframe with residuals, x value, y value

        Visualizes the predicted score against residuals
        """

        fig = px.scatter(df, x=x, y=y, color='score', color_continuous_scale=px.colors.sequential.Sunsetdark_r,
                        hover_data=['score', 'Division', 'Type'],
                        labels={
                                x: "Predicted Score",
                                "residuals": "Residuals"},
                        title="Residual Plot with Predicted Score")
        fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=5,
                y1=0,
                line=dict(color='Red',),
                xref='x',
                yref='y')
        fig.update_layout(font=dict(size=12), height=700)
        st.plotly_chart(fig, use_container_width=True)

    def plot_qq(df):
        """
        Inputs: dataframe with resiuals

        Visualizes a qq plot of the residuals to assess normality
        """
        
        z = (df.residuals-np.mean(df.residuals))/np.std(df.residuals)
        qqplot_data = qqplot(z, line='45').gca().lines

        fig = go.Figure()
        fig.add_trace({'type': 'scatter', 'x': qqplot_data[0].get_xdata(), 'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
            'marker': {'color': 'indianred'},
            'name': 'residuals'})

        fig.add_trace({'type': 'scatter', 'x': qqplot_data[1].get_xdata(), 'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {'color': 'Black'},
            'name': 'guide line'})

        fig['layout'].update({'title': 'Quantile-Quantile Plot',
            'xaxis': {'title': 'Theoritical Quantities', 'zeroline': False},
            'yaxis': {'title': 'Sample Quantities'},
            'showlegend': True,
            'width': 800,
            'height': 800})
        st.plotly_chart(fig)
                         
    def topnevals_bar(df, dimension, n = None):
        """
        Inputs: train/val dataframe

        Creates horizontal bar charts to see count of evaluations by a dimension and also creates histogram plot to see distribution of score
        """

        if dimension == 'score':
            bin_df = df.copy()
            bin_df['binned'] = bin_df[dimension].apply(lambda x: int(x))
            fig, ax = plt.subplots(figsize=(12,6))
            plt.rcParams.update({'font.size': 12})
            ax = sns.countplot(data=bin_df, x="binned")
            ax.set(xlabel="Score", title="Value Counts of Binned Score")
            st.pyplot(fig)
        else:
            sorted_dict = df[dimension].value_counts().to_dict()
            tup_list = sorted(sorted_dict.items(), key=lambda x:x[1], reverse=True)
            st.write("Total {}s found: {}".format(dimension, len(tup_list)))
            
            df = pd.DataFrame(tup_list[:n], columns=[dimension, 'count'])
            if n != None:
                t = f"Top {n} {dimension} by Count of Evaluations"
            else:
                t = f"All {dimension}s by Count of Evaluations"


            fig, ax = plt.subplots(figsize=(12,6))
            plt.rcParams.update({'font.size': 12})
            ax = sns.barplot(x="count", y=dimension, data=df, palette='GnBu_d')
            ax.set(xlabel="count", ylabel=dimension, title=t)
            st.pyplot(fig)

    def topscore_dim(df, dim):
        """
        Inputs: train/val dataframe, dimension

        Visualizes the top dimensions by course eval count to see how the score is distributed amongst them in a heatmap
        """

        bin_df = df.copy()
        if dim == 'Division':
            sorted_dict = bin_df[dim].value_counts().to_dict()
            tup_list = sorted(sorted_dict.items(), key=lambda x:x[1], reverse=True)
            top_divisions = [x[0] for x in tup_list[:10]]
            bin_df = bin_df[bin_df[dim].isin(top_divisions)]
        bin_df['binned'] = bin_df['score'].apply(lambda x: int(x))
        grouped = bin_df.groupby([dim, 'binned']).size().unstack(level=-1).fillna(0)
        fig, ax = plt.subplots(figsize=(12.5,12))
        plt.rcParams.update({'font.size': 12})
        ax = sns.heatmap(grouped, cmap = sns.cm.rocket_r, cbar_kws={'shrink':.9 }, linewidths=.3, linecolor='white')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set(xlabel="binned score", title="Heatmap of Scores by Top 10 {}s".format(dim))
        st.pyplot(fig)

    def num_dist(df, x):
        """
        Inputs: train/val dataframe or residuals df, x value

        Visualizes a histogram of values to see their distributions
        """

        if x == 'residuals':
            hover = ['score', 'Division', 'Type']
            fig = px.histogram(df, x=x, marginal="rug", color_discrete_sequence=['indianred'],
                            hover_data=hover,
                            labels={x: x},
                            title="Distribution of {}".format(x))
            fig.update_layout(font=dict(size=12), width=800, height=800)
            st.plotly_chart(fig)
        else:
            if x == 'score':
                fs = (12,6)
            else:
                fs = (25,6)
            fig, ax = plt.subplots(figsize=fs)
            plt.rcParams.update({'font.size': 12})
            ax = sns.histplot(data=df, x=x, kde=True)
            ax.set(xlabel=x, title="Distribution of {}".format(x))
            st.pyplot(fig)
    
    def correlation_heatmap(df):
        """
        Inputs: train/val dataframe

        Visualizes the numerical features in a heatmap to see correlated values and potential highest predictors
        """

        fig, ax = plt.subplots(figsize=(12.5,12))
        colormap = sns.diverging_palette(220, 10, as_cmap = True)
        plt.rcParams.update({'font.size': 12})
        ax = sns.heatmap(df.corr(), cmap=colormap, cbar_kws={'shrink':.9 }, annot=True, vmin=-1, vmax=1, linecolor='white', annot_kws={'fontsize':12 })
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set(title="Pearson Correlation of Features")
        st.pyplot(fig)
    
    def splomchart(df):
        """
        Inputs: train/val dataframe

        Visualizes a scatterplot matrix to see correlations and trends (linear/polynomial) amongst features and target variable
        """

        num_attribs = ['score', 'char_count', 'avg_word_length', 'avg_sentence_length', 'caps_vs_length']
        fig = sns.pairplot(df[num_attribs], diag_kind="kde", corner=True)
        st.pyplot(fig)

    def top_ngrams(df, n=10, tri = False):
        """
        Inputs: train/val dataframe, number of ngrams, whether to visualize trigrams

        Visualizes in a horizontal bar chart the top n ngrams
        """

        corpus = df['text_clean'].apply(lambda x: str(x).lower().split())
        lst_tokens = list(chain.from_iterable(corpus.values))

        if tri:
            fig, ax = plt.subplots(figsize=(25,6))
            ## trigrams
            dic_bi_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 3))
            trigrams = pd.DataFrame(dic_bi_freq.most_common(), columns=["Word","Freq"])
            trigrams["Word"] = trigrams["Word"].apply(lambda x: " ".join(string for string in x) )
            trigrams.set_index("Word").iloc[:n,:].sort_values(by="Freq") \
                .plot(kind="barh", title="Trigrams", ax=ax, legend=False) \
                    .grid(axis='x')
            ax.set(ylabel=None)
            st.pyplot(fig)
        else:            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,6))
            fig.suptitle("Most Frequent Words", fontsize=15)
            ## unigrams
            dic_uni_freq = nltk.FreqDist(lst_tokens)
            unigrams = pd.DataFrame(dic_uni_freq.most_common(), columns=["Word","Freq"])
            unigrams.set_index("Word").iloc[:n,:].sort_values(by="Freq") \
                .plot(kind="barh", title="Unigrams", ax=ax[0], legend=False) \
                    .grid(axis='x')
            ax[0].set(ylabel=None)
                
            ## bigrams
            dic_bi_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
            bigrams = pd.DataFrame(dic_bi_freq.most_common(), columns=["Word","Freq"])
            bigrams["Word"] = bigrams["Word"].apply(lambda x: " ".join(string for string in x) )
            bigrams.set_index("Word").iloc[:n,:].sort_values(by="Freq") \
                .plot(kind="barh", title="Bigrams", ax=ax[1], legend=False) \
                    .grid(axis='x')
            ax[1].set(ylabel=None)
            st.pyplot(fig)   


    def feature_imp(coef, transformer, n, onlytop = False):
        """
        Inputs: coeficient values after prediction, top n features, whether to only show most significant features or not

        Visualizes a horizontal bar chart that identifies top (and potentially bottom) n features and their coefficients
        """

        feature_importances = coef
        num_attribs = ['char_count', 'avg_word_length', 'avg_sentence_length', 'caps_vs_length']
        cat_encoder = transformer.transformers_[1][1]
        txt_encoder = transformer.transformers_[2][1]

        cat_one_hot_attribs = list(cat_encoder.get_feature_names())
        txt_vect_attribs = list(txt_encoder.get_feature_names())
        attributes = num_attribs + cat_one_hot_attribs + txt_vect_attribs
        feat_imps = sorted(zip(feature_importances, attributes), reverse=True)
        if onlytop:
            topn = feat_imps[:n*2]
            feat_imps_df = pd.DataFrame(topn, columns=['coeficients', 'features'])       
        else:
            topn = feat_imps[:n]
            botn = feat_imps[-n:]
            feat_imps_df = pd.DataFrame(topn + botn, columns=['coeficients', 'features'])

        fig = px.bar(feat_imps_df, x='coeficients', y='features', orientation='h', color='coeficients', color_continuous_scale=px.colors.diverging.Temps_r,
                    title="Top/Bottom {} Feature Importances".format(n))
        fig.update_layout(font=dict(size=12), height=800, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)    

    def plot_top_words(model, word_ids, corpus, n_topics, n_top_words, title):
        """
        Inputs: lda/nmf model, word to id map, corpus of text, number of topics, number of top words per topic to show, and title of chart

        Visualizes multiple horizontal bar charts indicating the n topic models that result and their m top words for each with the coherence score per topic 
        """

        fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True)
        axes = axes.flatten()
        for topic_idx in range(0,n_topics):
            topic = model.get_topic_terms(topic_idx)
            top_features_ind = [x[0] for x in topic]
            top_features = [word_ids[x] for x in top_features_ind]
            weights = [x[1] for x in topic]
            coherence_model_lda = gensim.models.CoherenceModel(model=model, texts=corpus, dictionary=word_ids, coherence='c_v')
            topic_coherence = coherence_model_lda.get_coherence_per_topic()[topic_idx]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx +1}, Coherence: {round(topic_coherence, 2)}',
                        fontdict={'fontsize': 16})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=12)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=16)

        plt.subplots_adjust(top=0.90, wspace=0.90, hspace=0.3)
        st.pyplot(fig)
    
    train_val_df, test_df, eval_demog_df = load_data()

    y = train_val_df.score
    x = train_val_df.drop(columns=['score', 'Comment.Text.Processed'])
    test_fin = test_df.drop(columns=['score', 'Comment.Text.Processed']).dropna()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

    if st.sidebar.button("Run Exploratory Data Analysis", key='run_eda'):
        st.subheader("Exploratory Data Analysis")
        col1, col2 = st.beta_columns(2)
        
        with col1:
            topnevals_bar(train_val_df, 'Division', 10)
            topscore_dim(train_val_df, 'Division')
            topnevals_bar(train_val_df, 'score')
            correlation_heatmap(train_val_df)
        with col2:
            topnevals_bar(train_val_df, 'Type')
            topscore_dim(train_val_df, 'Type')
            num_dist(train_val_df, 'score')

            splomchart(train_val_df)

        top_ngrams(train_val_df, 20)
        top_ngrams(train_val_df, 20, tri=True)

    st.sidebar.subheader("Choose Regressor")
    st.sidebar.markdown("OLS and SVM are run through the SGDRegressor due to a large dataset.")
    regressor = st.sidebar.selectbox("Regressor", ("Linear Regression (OLS)", "Support Vector Machine (SVM)", "Gradient Boosting", "Random Forest", "XGBoost"))

    if regressor == 'Linear Regression (OLS)':
        st.sidebar.subheader("Model Hyperparameters")
        l1_ratio = st.sidebar.number_input("Ratio of L1 (value of 1)/L2 (value of 0) Regularization", 0.0, 1.0, step=0.1, key='L1_L2')
        alpha = st.sidebar.select_slider("Constant that multiplies the regularization term", options=list(10.0**-np.arange(1,7)), key='alpha')
        max_iter = st.sidebar.number_input("The maximum number of passes over the training data (aka epochs)", 100, 5000, step=100, key='max_iter')
        early_stopping = st.sidebar.radio("Whether to use early stopping to terminate training when validation score is not improving", ('True', 'False'), key='estop')
        if early_stopping == 'True':
            early_stopping = True
            validation_fraction = st.sidebar.number_input("The proportion of training data to set aside as validation set for early stopping", 0.0, 1.0, step=0.1, key='val_frac')
        else:
            early_stopping = False
        st.sidebar.subheader("Additional Options")
        train_subset = st.sidebar.checkbox("Train on Subset of Data", False, key="error")
        error_analysis = st.sidebar.checkbox("Conduct Error Analysis", False, key="error")
        download_preds = st.sidebar.checkbox("Download Final Predictions to CSV", False, key="dload")

        if st.sidebar.button("Run Model", key='run'):
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("OLS Regression (with Stochastic Gradient Descent) Results")
                model = SGDRegressor(penalty='elasticnet', learning_rate='adaptive', validation_fraction=validation_fraction, 
                                    random_state=RANDOM_SEED, shuffle=True, l1_ratio=l1_ratio, alpha=alpha, max_iter=max_iter, early_stopping=early_stopping)
                if train_subset:
                    X_train = X_train[:7500]
                    y_train = y_train[:7500]

                model_fitted, y_pred, final_pred, col_transform = train_model(model, X_train, X_test, y_train, y_test, test_fin)

                st.write('Mean Squared Error: ', mean_squared_error(y_test, y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, y_pred).round(2))
                preds_df = pd.DataFrame(y_pred, columns=['pred_score'])
                ytest_df = pd.DataFrame(y_test, columns=['score'])
                predtest_df = ytest_df.join(preds_df)
                residualsdf = x.join(predtest_df, how='right')
                plot_metrics('score', 'pred_score', residualsdf)
            with col2:
                st.subheader("Dummy Regression (mean strategy) Results")
                dummy_regr = DummyRegressor(strategy='mean')
                
                _, dummy_y_pred, _, _ = train_model(dummy_regr, X_train, X_test, y_train, y_test, test_fin)
                
                st.write('Mean Squared Error: ', mean_squared_error(y_test, dummy_y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, dummy_y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, dummy_y_pred).round(2))
                plot_metrics('score', 'pred_score', dummy=True, ytest=y_test, ypred=dummy_y_pred)
            
            feature_imp(model_fitted.regressor_.coef_, col_transform, 20)

            if error_analysis:
                st.subheader("Error Analysis")
                residualsdf['residuals'] = residualsdf['score'] - residualsdf['pred_score']
                n = 100
                st.write('Top {} Course Evaluations that had the highest error'.format(n))
                residualsdf = residualsdf.reindex(residualsdf.residuals.abs().sort_values(ascending=False).index)
                st.dataframe(residualsdf[:n])
                plot_residuals(residualsdf, 'pred_score', 'residuals')
                squared_errors = (y_pred - y_test) ** 2
                c1, c2 = st.beta_columns(2)
                with c1:
                    num_dist(residualsdf, 'residuals')
                with c2:
                    plot_qq(residualsdf)
                
            if download_preds:
                preds_df = pd.DataFrame(list(zip(test_df.index, final_pred)), columns = ['id', 'score'])
                open('{} Predictions.csv'.format(regressor), 'w', newline='').write(preds_df.to_csv(index=False))

    if regressor == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        l1_ratio = st.sidebar.number_input("Ratio of L1 (value of 1)/L2 (value of 0) Regularization", 0.0, 1.0, step=0.1, key='L1_L2')
        alpha = st.sidebar.select_slider("Constant that multiplies the regularization term", options=list(10.0**-np.arange(1,7)), key='alpha')
        max_iter = st.sidebar.number_input("The maximum number of passes over the training data (aka epochs)", 100, 5000, step=100, key='max_iter')
        early_stopping = st.sidebar.radio("Whether to use early stopping to terminate training when validation score is not improving", ('True', 'False'), key='estop')
        if early_stopping == 'True':
            early_stopping = True
            validation_fraction = st.sidebar.number_input("The proportion of training data to set aside as validation set for early stopping.", 0.0, 1.0, step=0.1, key='val_frac')
        else:
            early_stopping = False
        st.sidebar.subheader("Additional Options")
        train_subset = st.sidebar.checkbox("Train on Subset of Data", False, key="error")
        error_analysis = st.sidebar.checkbox("Conduct Error Analysis", False, key="error")
        download_preds = st.sidebar.checkbox("Download Final Predictions to CSV", False, key="dload")

        if st.sidebar.button("Run Model", key='run'):
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("Support Vector Regression (with Stochastic Gradient Descent) Results")
                model = SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet', learning_rate='adaptive', validation_fraction=validation_fraction, 
                                    random_state=RANDOM_SEED, shuffle=True, l1_ratio=l1_ratio, alpha=alpha, max_iter=max_iter, early_stopping=early_stopping)
                if train_subset:
                    X_train = X_train[:7500]
                    y_train = y_train[:7500]

                model_fitted, y_pred, final_pred, col_transform = train_model(model, X_train, X_test, y_train, y_test, test_fin)

                st.write('Mean Squared Error: ', mean_squared_error(y_test, y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, y_pred).round(2))
                preds_df = pd.DataFrame(y_pred, columns=['pred_score'])
                ytest_df = pd.DataFrame(y_test, columns=['score'])
                predtest_df = ytest_df.join(preds_df)
                residualsdf = x.join(predtest_df, how='right')
                plot_metrics('score', 'pred_score', residualsdf)
            with col2:
                st.subheader("Dummy Regression (mean strategy) Results")
                dummy_regr = DummyRegressor(strategy='mean')
                
                _, dummy_y_pred, _, _ = train_model(dummy_regr, X_train, X_test, y_train, y_test, test_fin)
                
                st.write('Mean Squared Error: ', mean_squared_error(y_test, dummy_y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, dummy_y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, dummy_y_pred).round(2))
                plot_metrics('score', 'pred_score', dummy=True, ytest=y_test, ypred=dummy_y_pred)
            
            feature_imp(model_fitted.regressor_.coef_, col_transform, 20)

            if error_analysis:
                st.subheader("Error Analysis")
                residualsdf['residuals'] = residualsdf['score'] - residualsdf['pred_score']
                n = 100
                st.write('Top {} Course Evaluations that had the highest error'.format(n))
                residualsdf = residualsdf.reindex(residualsdf.residuals.abs().sort_values(ascending=False).index)
                st.dataframe(residualsdf[:n])
                plot_residuals(residualsdf, 'pred_score', 'residuals')
                squared_errors = (y_pred - y_test) ** 2
                c1, c2 = st.beta_columns(2)
                with c1:
                    num_dist(residualsdf, 'residuals')
                with c2:
                    plot_qq(residualsdf)
                
            if download_preds:
                preds_df = pd.DataFrame(list(zip(test_df.index, final_pred)), columns = ['id', 'score'])
                open('{} Predictions.csv'.format(regressor), 'w', newline='').write(preds_df.to_csv(index=False))

    if regressor == 'Gradient Boosting':
        st.sidebar.subheader("Model Hyperparameters")
        learning_rate = st.sidebar.number_input("The learning rate shrinks the contribution of each tree", .01, .10, step=.01, key='learning_rate')
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        subsample = st.sidebar.number_input("The fraction of samples to be used for fitting the individual base learners", .10, 1.00, step=.10, key='learning_rate')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
        max_features = st.sidebar.selectbox("The number of features to consider when looking for best split", ('auto', 'sqrt', 'log2'), 0, key='max_feats')

        st.sidebar.subheader("Additional Options")
        train_subset = st.sidebar.checkbox("Train on Subset of Data", False, key="error")
        error_analysis = st.sidebar.checkbox("Conduct Error Analysis", False, key="error")        
        download_preds = st.sidebar.checkbox("Download Final Predictions to CSV", False, key="dload")

        if st.sidebar.button("Run Model", key='run'):
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("Gradient Boosting Regression Results")
                model = GradientBoostingRegressor(loss='ls', learning_rate=learning_rate, subsample=subsample,
                                                max_depth=max_depth, max_features=max_features, warm_start=warm_start, random_state=RANDOM_SEED)
                if train_subset:
                    X_train = X_train[:7500]
                    y_train = y_train[:7500]

                model_fitted, y_pred, final_pred, col_transform = train_model(model, X_train, X_test, y_train, y_test, test_fin)

                st.write('Mean Squared Error: ', mean_squared_error(y_test, y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, y_pred).round(2))
                preds_df = pd.DataFrame(y_pred, columns=['pred_score'])
                ytest_df = pd.DataFrame(y_test, columns=['score'])
                predtest_df = ytest_df.join(preds_df)
                residualsdf = x.join(predtest_df, how='right')
                plot_metrics('score', 'pred_score', residualsdf)
            with col2:
                st.subheader("Dummy Regression (mean strategy) Results")
                dummy_regr = DummyRegressor(strategy='mean')
                
                _, dummy_y_pred, _, _ = train_model(dummy_regr, X_train, X_test, y_train, y_test, test_fin)
                
                st.write('Mean Squared Error: ', mean_squared_error(y_test, dummy_y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, dummy_y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, dummy_y_pred).round(2))
                plot_metrics('score', 'pred_score', dummy=True, ytest=y_test, ypred=dummy_y_pred)
            
            feature_imp(model_fitted.regressor_.coef_, col_transform, 20)

            if error_analysis:
                st.subheader("Error Analysis")
                residualsdf['residuals'] = residualsdf['score'] - residualsdf['pred_score']
                n = 100
                st.write('Top {} Course Evaluations that had the highest error'.format(n))
                residualsdf = residualsdf.reindex(residualsdf.residuals.abs().sort_values(ascending=False).index)
                st.dataframe(residualsdf[:n])
                plot_residuals(residualsdf, 'pred_score', 'residuals')
                squared_errors = (y_pred - y_test) ** 2
                c1, c2 = st.beta_columns(2)
                with c1:
                    num_dist(residualsdf, 'residuals')
                with c2:
                    plot_qq(residualsdf)
                
            if download_preds:
                preds_df = pd.DataFrame(list(zip(test_df.index, final_pred)), columns = ['id', 'score'])
                open('{} Predictions.csv'.format(regressor), 'w', newline='').write(preds_df.to_csv(index=False))

    if regressor == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
        max_features = st.sidebar.selectbox("The number of features to consider when looking for best split", ('auto', 'sqrt', 'log2'), 0, key='max_feats')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        st.sidebar.subheader("Additional Options")
        train_subset = st.sidebar.checkbox("Train on Subset of Data", False, key="error")
        error_analysis = st.sidebar.checkbox("Conduct Error Analysis", False, key="error")        
        download_preds = st.sidebar.checkbox("Download Final Predictions to CSV", False, key="dload")

        if st.sidebar.button("Run Model", key='run'):
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("Random Forest Regression Results")
                model = RandomForestRegressor(max_depth=max_depth, max_features=max_features, bootstrap=bootstrap)
                if train_subset:
                    X_train = X_train[:7500]
                    y_train = y_train[:7500]

                model_fitted, y_pred, final_pred, col_transform = train_model(model, X_train, X_test, y_train, y_test, test_fin)

                st.write('Mean Squared Error: ', mean_squared_error(y_test, y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, y_pred).round(2))
                preds_df = pd.DataFrame(y_pred, columns=['pred_score'])
                ytest_df = pd.DataFrame(y_test, columns=['score'])
                predtest_df = ytest_df.join(preds_df)
                residualsdf = x.join(predtest_df, how='right')
                plot_metrics('score', 'pred_score', residualsdf)
            with col2:
                st.subheader("Dummy Regression (mean strategy) Results")
                dummy_regr = DummyRegressor(strategy='mean')
                
                _, dummy_y_pred, _, _ = train_model(dummy_regr, X_train, X_test, y_train, y_test, test_fin)
                
                st.write('Mean Squared Error: ', mean_squared_error(y_test, dummy_y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, dummy_y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, dummy_y_pred).round(2))
                plot_metrics('score', 'pred_score', dummy=True, ytest=y_test, ypred=dummy_y_pred)
            
            feature_imp(model_fitted.regressor_.coef_, col_transform, 20)

            if error_analysis:
                st.subheader("Error Analysis")
                residualsdf['residuals'] = residualsdf['score'] - residualsdf['pred_score']
                n = 100
                st.write('Top {} Course Evaluations that had the highest error'.format(n))
                residualsdf = residualsdf.reindex(residualsdf.residuals.abs().sort_values(ascending=False).index)
                st.dataframe(residualsdf[:n])
                plot_residuals(residualsdf, 'pred_score', 'residuals')
                squared_errors = (y_pred - y_test) ** 2
                c1, c2 = st.beta_columns(2)
                with c1:
                    num_dist(residualsdf, 'residuals')
                with c2:
                    plot_qq(residualsdf)
                
            if download_preds:
                preds_df = pd.DataFrame(list(zip(test_df.index, final_pred)), columns = ['id', 'score'])
                open('{} Predictions.csv'.format(regressor), 'w', newline='').write(preds_df.to_csv(index=False))

    if regressor == 'XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        eta = st.sidebar.number_input("Step size shrinkage used in update to prevents overfitting (learning rate)", 0.00, 1.00, step=0.05, key='eta')
        subsample = st.sidebar.number_input("Subsample ratio of the training instances", 0.0, 1.0, step=0.1, key='subsamp')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
        reg_lambda = st.sidebar.number_input("L2 regularization term on weights", 0.0, 1.0, step=0.1, key='lambda')
        reg_alpha = st.sidebar.number_input("L1 regularization term on weights", 0.0, 1.0, step=0.1, key='alpha')

        st.sidebar.subheader("Additional Options")
        train_subset = st.sidebar.checkbox("Train on Subset of Data", False, key="error")
        error_analysis = st.sidebar.checkbox("Conduct Error Analysis", False, key="error")
        download_preds = st.sidebar.checkbox("Download Final Predictions to CSV", False, key="dload")

        if st.sidebar.button("Run Model", key='run'):
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("eXtreme Gradient Boosting Regression Results")
                model = xg.XGBRegressor(seed=RANDOM_SEED, objective='reg:squarederror', eta=eta, subsample=subsample, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

                if train_subset:
                    X_train = X_train[:7500]
                    y_train = y_train[:7500]

                model_fitted, y_pred, final_pred, col_transform = train_model(model, X_train, X_test, y_train, y_test, test_fin)

                st.write('Mean Squared Error: ', mean_squared_error(y_test, y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, y_pred).round(2))
                preds_df = pd.DataFrame(y_pred, columns=['pred_score'])
                ytest_df = pd.DataFrame(y_test, columns=['score'])
                predtest_df = ytest_df.join(preds_df)
                residualsdf = x.join(predtest_df, how='right')
                plot_metrics('score', 'pred_score', residualsdf)
            with col2:
                st.subheader("Dummy Regression (mean strategy) Results")
                dummy_regr = DummyRegressor(strategy='mean')
                
                _, dummy_y_pred, _, _ = train_model(dummy_regr, X_train, X_test, y_train, y_test, test_fin)
                
                st.write('Mean Squared Error: ', mean_squared_error(y_test, dummy_y_pred).round(4))
                st.write('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, dummy_y_pred)).round(4))
                st.write('Coefficient of Determination: ', r2_score(y_test, dummy_y_pred).round(2))
                plot_metrics('score', 'pred_score', dummy=True, ytest=y_test, ypred=dummy_y_pred)
            
            feature_imp(model_fitted.regressor_.coef_, col_transform, 20)

            if error_analysis:
                st.subheader("Error Analysis")
                residualsdf['residuals'] = residualsdf['score'] - residualsdf['pred_score']
                n = 100
                st.write('Top {} Course Evaluations that had the highest error'.format(n))
                residualsdf = residualsdf.reindex(residualsdf.residuals.abs().sort_values(ascending=False).index)
                st.dataframe(residualsdf[:n])
                plot_residuals(residualsdf, 'pred_score', 'residuals')
                squared_errors = (y_pred - y_test) ** 2
                c1, c2 = st.beta_columns(2)
                with c1:
                    num_dist(residualsdf, 'residuals')
                with c2:
                    plot_qq(residualsdf)
                
            if download_preds:
                preds_df = pd.DataFrame(list(zip(test_df.index, final_pred)), columns = ['id', 'score'])
                open('{} Predictions.csv'.format(regressor), 'w', newline='').write(preds_df.to_csv(index=False))


    if st.sidebar.checkbox("Show subset of raw training data", False):
        st.subheader("Course Evaluation Training Data Set (Regression)")
        st.dataframe(train_val_df.sample(frac=0.5, replace=True, random_state=RANDOM_SEED))
        st.markdown("This [data set](https://www.kaggle.com/c/umich-siads-695-fall20-predict-course-eval-scores) includes 50% random sampling (with replacement) of the training csv file for course evaluations. "
        "It contains the Division of UM that the class resides in, the Type of course it is, and a free text response of the student's evaluation. It also contains the target variable "
        "of score that is the student's evaluation score of the course.")
    
    st.sidebar.title("Unsupervised Learning")
    st.sidebar.markdown("Unsupervised Learning Approaches to Explore Demographics")

    st.sidebar.subheader("Choose Topic Model")
    topicmodel = st.sidebar.selectbox("Topic Model", ("Latent Dirichlet Allocation (LDA)", "Non-Negative Matrix Factorization (NMF)"))

    if topicmodel == 'Latent Dirichlet Allocation (LDA)':
        n_components = st.sidebar.number_input("The number of topics", 2, 10, step=1, key='n_components')
        n_grams = st.sidebar.number_input("N-grams to use", 1, 3, step=1, key='ngrams')
        chunksize = st.sidebar.number_input("Number of documents to be used in each training chunk", 100, 2000, step=100, key='chunksize')
        passes = st.sidebar.number_input(" Number of passes through the corpus during training", 1, 20, step=1, key='passes')
        topn = st.sidebar.number_input("Number of the most significant words that are associated with the topic", 5, 20, step=1, key='topn')

        if st.sidebar.button("Run Model", key='run_unsup'):
            corpus = eval_demog_df["text_clean"]

            ## pre-process corpus
            lst_corpus = []
            for string in corpus:
                lst_words = string.split()
                lst_grams = [" ".join(lst_words[i:i + 2]) for i in range(0, len(lst_words), n_grams)]
                lst_corpus.append(lst_grams)
            ## map words to an id
            word_id_map = gensim.corpora.Dictionary(lst_corpus)
            ## create dictionary word:freq
            corpus_dict = [word_id_map.doc2bow(word) for word in lst_corpus]

            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_dict, id2word=word_id_map, num_topics=n_components, random_state=RANDOM_SEED, 
                                                        chunksize=chunksize, alpha='auto', eta='auto', passes=passes, eval_every=1)
            plot_top_words(lda_model, word_id_map, lst_corpus, n_components, topn, 'Topics in LDA model')

    if topicmodel == 'Non-Negative Matrix Factorization (NMF)':
        n_components = st.sidebar.number_input("The number of topics", 2, 10, step=1, key='n_components')
        n_grams = st.sidebar.number_input("N-grams to use", 1, 3, step=1, key='ngrams')
        chunksize = st.sidebar.number_input("Number of documents to be used in each training chunk", 100, 2000, step=100, key='chunksize')
        passes = st.sidebar.number_input(" Number of passes through the corpus during training", 1, 20, step=1, key='passes')
        topn = st.sidebar.number_input("Number of the most significant words that are associated with the topic", 5, 20, step=1, key='topn')

        if st.sidebar.button("Run Model", key='run_unsup'):
            corpus = eval_demog_df["text_clean"]

            ## pre-process corpus
            lst_corpus = []
            for string in corpus:
                lst_words = string.split()
                lst_grams = [" ".join(lst_words[i:i + 2]) for i in range(0, len(lst_words), n_grams)]
                lst_corpus.append(lst_grams)
            ## map words to an id
            word_id_map = gensim.corpora.Dictionary(lst_corpus)
            ## create dictionary word:freq
            corpus_dict = [word_id_map.doc2bow(word) for word in lst_corpus]

            nmf_model = gensim.models.nmf.Nmf(corpus=corpus_dict, id2word=word_id_map, num_topics=n_components, random_state=RANDOM_SEED, 
                                                chunksize=chunksize, passes=passes, eval_every=1)
            plot_top_words(nmf_model, word_id_map, lst_corpus, n_components, topn, 'Topics in NMF model')

    if st.sidebar.checkbox("Show subset of raw demographic data", False):
        st.subheader("Course Evaluation with Demographics Data Set")
        st.dataframe(eval_demog_df.sample(frac=0.5, replace=True, random_state=RANDOM_SEED))
        st.markdown("This [data set](https://www.kaggle.com/c/umich-siads-695-fall20-predict-course-eval-scores) includes 50% random sampling (with replacement) of the course evaluation with demographics csv. "
        "It contains the Division of UM that the class resides in, the Type of course it is, the title/gender/tenure of the instructor, and a free text response of the student's evaluation.")

if __name__ == '__main__':
    main()