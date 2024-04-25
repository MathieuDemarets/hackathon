# Regular imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import datetime
import sys

# Multi-threading scraping
from bs4 import BeautifulSoup
from selenium import webdriver
from itertools import repeat
import concurrent.futures
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Geospatial processing
import geopandas as gpd
import folium

# Network analysis
import pickle
import networkx as nx

# Statistical analysis
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm

# ML Regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor

# ML Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ML Helpers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split

pd.options.display.max_columns = None


class MDO_Researcher():
    """Core member of the Movie Director Office, the researcher will try to collect and clean the data.
    The data will be used by the other members for exploration, visualization, and prediction.

    Attributes
    ----------
    pre : pd.DataFrame
        Pre-release dataset
    after : pd.DataFrame
        After-release dataset
    full : pd.DataFrame
        Merged dataset
    pred_columns : list
        List of predictors
    success_columns : list
        List of success metrics

    Initiation
    ----------

    from_directory(directory:str) -> None
        Create a MDO_Researcher object from a directory containing the pre and after release datasets.

    from_enriched(
        path:str="mdo/data/enriched_data.csv",
        successes:list=[
            "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "movie_facebook_likes",
            "profit", "num_critic_for_reviews"],
            **class_arg) -> None
        Initiate the dataset from a csv file with enriched data.


    Methods
    -------

    prepare_data(verbose:bool=False) -> None
        Prepare the data by dropping duplicates and merging the two datasets.

    split_genres(top:int=10, verbose:bool=False) -> None
        Split the top genres into individual columns and add one for all other genres.

    enrich_with_country(
        path:str="mdo/data/countries_shape/ne_110m_admin_0_countries.shp",
        country_replacement:dict={"United States of America": "USA", "United Kingdom": "UK", "Czechia": "Czech Republic"},
        columns:list=["NAME", "POP_EST", "GDP_MD"], replace:bool=True) -> None
        Enrich the data with information about the country of the movie.

    enrich_with_actors(
        path:str="mdo/data/actors.pickle",
        centralities:list=["degree", "closeness", "betweenness"], replace:bool=True) -> None
        Enrich the data with information about the network of actors across movies.

    research_release_years(
        movie_titles:list=[], save:bool=False, path:str="mdo/data/release_years.csv", workers:int=5) -> pd.DataFrame
        Research the release years of the movies in the dataset through webscraping.

    enrich_with_release_year(path:str="mdo/data/release_years.csv", replace:bool=True) -> pd.DataFrame
        Enrich the dataset with the release years of the movies.

    enrich_all(
        save:bool=False, path:str="mdo/data/enriched_dataset.csv", country:bool=True, actors:bool=True,
        release_year:bool=True, country_par:dict={}, actors_par:dict={}, release_par:dict={}) -> pd.DataFrame
        Enrich the data with all the available methods.
    """

    def __init__(
            self,
            pre: str = "mdo/data/pre_release.csv",
            after: str = "mdo/data/after_release.csv") -> None:

        self.pre = pd.read_csv(pre)
        self.after = pd.read_csv(after)
        self.full = None
        self.pred_columns = None
        self.success_columns = None

    # Automatic generation:

    @classmethod
    def from_directory(cls, directory: str, drop_na: bool = True) -> None:
        """Create a MovieCFO object from a directory containing the pre and after release datasets.

        Parameters
        ----------
        directory : str
            Path to the directory containing the two datasets
        """
        for file in os.listdir(directory):
            if "pre" in file and file.endswith(".csv"):
                pre = os.path.join(directory, file)
            elif "after" in file and file.endswith(".csv"):
                after = os.path.join(directory, file)
        obj = cls(pre, after)
        obj.prepare_data()
        obj.full.dropna(inplace=drop_na)
        return obj

    # Initiate class with enriched data

    @classmethod
    def from_enriched(
            cls,
            path: str = "mdo/data/enriched_data.csv",
            successes=[
                "gross", "num_voted_users", "num_user_for_reviews",
                "imdb_score", "movie_facebook_likes", "profit", "num_critic_for_reviews"],
            drop_na: bool = False,
            **class_arg) -> None:
        """Initiate the dataset from a csv file with enriched data.

        Parameters
        ----------
        path : str, optional
            Path to the enriched dataset, by default "mdo/data/enriched_data.csv"
        successes : list, optional
            Columns which are observed after the movie release.
        drop_na : bool, optional
            Drop missing values, by default False
        class_arg : dict
            Arguments for the class initiation (pre and after datasets)
        """
        obj = cls(**class_arg)
        obj.full = pd.read_csv(path)
        obj.success_columns = successes
        obj.pred_columns = [
            col for col in obj.full.columns if col not in obj.success_columns]
        if drop_na:
            obj.full.dropna(inplace=True)
        return obj

    # ># Method 1:

    def prepare_data(self, verbose: bool = False) -> None:
        """Prepare the data by dropping duplicates and merging the two datasets.

        Parameters
        ----------
        verbose : bool, optional
            Display execution details, by default False
        """
        # ---
        if verbose:
            print("".join([
                f"Pre-release: {self.pre.shape[0]} rows with {
                    self.pre.movie_title.unique().shape[0]}",
                " unique movies\n",
                f"After-release: {self.after.shape[0]} rows with {
                    self.after.movie_title.unique().shape[0]}",
                " unique movies\n"
            ]))

        # Dropping duplicates before merging
        self.pre.drop_duplicates(subset=["movie_title"], inplace=True)
        self.after.drop_duplicates(subset=["movie_title"], inplace=True)

        # Merging the two datasets
        self.full = pd.merge(
            self.pre, self.after, on="movie_title",
            how="inner", suffixes=("_pred", "_success"))
        self.pred_columns = [col for col in self.pre.columns]
        self.success_columns = [
            col for col in self.after.columns if col != "movie_title"]

        # ---
        if verbose:
            print(f"Full dataset: {self.full.shape[0]} unique movies.")

    # ># Method 2:

    def split_genres(self, top: int = 10, verbose: bool = False) -> None:
        """Split the genres into individual columns.

        Parameters
        ----------
        top : int, optional
            Number of top genres to keep, by default 10
        verbose : bool, optional
            Display execution details, by default False
        """

        # Get the top genres and the rest
        self.full["genres"] = self.full["genres"].str.split("|")
        genre_count = self.full["genres"].explode().value_counts()
        top_genres = genre_count.head(top).index.tolist()
        other_genres = genre_count.index.difference(top_genres).tolist()

        if verbose:
            print(f"Top genres:")
            print(genre_count.head(top))

        # Create the columns
        if not self.full.columns.str.contains("genre_").any():
            for genre in top_genres:
                self.full[f"genre_{genre}"] = self.full["genres"].apply(
                    lambda x: genre in x).astype(int)
            self.full["genre_Other"] = self.full["genres"].apply(
                lambda x: set(x).intersection(other_genres) != set()).astype(int)

        # Update the list of predictors
        self.pred_columns += [
            f"genre_{genre}" for genre in top_genres] + ["genre_Other"]

    # ># Method 3:

    def enrich_with_country(
        self,
        path: str = "mdo/data/countries_shape/ne_110m_admin_0_countries.shp",
        country_replacement: dict = {
            "United States of America": "USA",
            "United Kingdom": "UK",
            "Czechia": "Czech Republic"
        },
        columns: list = ["NAME", "POP_EST", "GDP_MD"],
        replace: bool = True
    ) -> None:
        """
        Enrich the data with information about the country of the movie.

        Parameters
        ----------
        path : str, optional
            Path to the shapefile containing the countries, by default "mdo/data/countries_shape/ne_110m_admin_0_countries.shp"
        country_replacement : dict, optional
            Dictionary to replace country names, by default {"United States of America": "USA", "United Kingdom": "UK", "Czechia": "Czech Republic"}
        columns : list, optional
            Columns to keep from the shapefile, by default ["NAME", "POP_EST", "GDP_MD"] (NAME needs to be present)
        replace : bool, optional
            Replace the full dataset with the enriched one, by default True

        Returns
        -------
        pd.DataFrame
            Enriched dataset

        Raises
        ------
        KeyError
            If the chosen columns are not available in the dataset
        FileNotFoundError
            If the file is not found      
        """

        # Load the shapefile and select the columns
        try:
            countries_mask = gpd.read_file(path)
        except FileNotFoundError:
            raise FileNotFoundError("The countries shapefile was not found")
        try:
            countries_mask = countries_mask[columns]
            countries_mask.columns = [col.lower() for col in columns]
        except KeyError:
            raise KeyError(
                "The chosen columns are not available in the dataset")

        # Rename the countries and merge the datasets
        for key in country_replacement.keys():
            countries_mask.name = countries_mask.name.str.replace(
                key, country_replacement[key])
        enriched_data = self.full.merge(
            countries_mask, left_on="country", right_on="name", how="left")
        enriched_data.drop(columns="name", inplace=True)

        if replace:
            self.full = enriched_data
            self.pred_columns += [col for col in countries_mask.columns if col != "name"]

        return enriched_data

    # ># Method 4:

    def enrich_with_actors(
            self,
            path: str = "mdo/data/actors.pickle",
            centralities: list = ["degree", "closeness", "betweenness"],
            replace: bool = True) -> None:
        """Enrich the data with information about the network of actors across movies.
        It requires the actors agent to generate the network first.

        Parameters
        ----------
        path : str, optional
            Path to the actors network, by default "mdo/data/actors.pickle"
        centralities : list, optional
            Centralities to add to the dataset, by default "degree", "closeness", and "betweenness"
        replace : bool, optional
            Replace the full dataset with the enriched one, by default True

        Raises
        ------
        FileNotFoundError
            If the file is not found

        Returns
        -------
        pd.DataFrame
            Enriched dataset
        """

        # Get the graph of actors
        try:
            G = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            raise FileNotFoundError("The actors network file was not found")

        # Get the actors data from the graph
        actors_data = pd.DataFrame(G.nodes())
        actors_data.columns = ["actor_name"]

        if "degree" in centralities:
            dc = pd.DataFrame(G.degree(), columns=["actor_name", "degree"])
            actors_data = actors_data.merge(dc, on="actor_name", how="left")
        if "closeness" in centralities:
            cc = pd.DataFrame(nx.closeness_centrality(
                G).items(), columns=["actor_name", "closeness"])
            actors_data = actors_data.merge(cc, on="actor_name", how="left")
        if "betweenness" in centralities:
            bc = pd.DataFrame(nx.betweenness_centrality(
                G).items(), columns=["actor_name", "betweenness"])
            actors_data = actors_data.merge(bc, on="actor_name", how="left")

        # Enrich the dataset for the three actors
        enriched_data = self.full.copy()
        for i in range(1, 4):
            temp_act_data = actors_data.copy()
            temp_act_data.columns = [col if col == "actor_name" else f"actor_{
                i}_{col}" for col in temp_act_data.columns]
            enriched_data = enriched_data.merge(temp_act_data, left_on=f"actor_{
                                                i}_name", right_on="actor_name", how="left")
            enriched_data.drop(columns="actor_name", inplace=True)

        if replace:
            self.full = enriched_data
            for i in range(1, 4):
                self.pred_columns += [f"actor_{i}_{
                    col}" for col in actors_data.columns if col != "actor_name"]

        return enriched_data

    # ># Method 5:

    def research_release_years(
            self, movie_titles: list = [], save: bool = False,
            path: str = "mdo/data/release_years.csv", workers: int = 5) -> pd.DataFrame:
        """Research the release years of the movies in the dataset through webscraping.
        We fetch the release date from IMDB with 'workers' Chrome drivers running in parallel.

        Parameters
        ----------
        movie_titles : list, optional
            List of titles to research, by default [] and will research the 10 first movies
            from self.full
        save : bool, optional
            Save the release year DataFrame to a file, by default False
        path : str, optional
            Path to save the release years DataFrame, by default "mdo/data/release_years.csv"
        workers : int, optional
            Number of parallel workers for the scraping, by default 5

        Return
        ------
        pd.DataFrame
            DataFrame with the release years of the movies

        Legal Notice
        ------------
        [Non-commercial use only](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX?ref_=helpart_nav_18#)
            Information courtesy of IMDb
            (https://www.imdb.com).
            Used with permission.
        """

        # Opening a single Chrome driver
        def open_driver(url):
            options = webdriver.ChromeOptions()
            options.experimental_options["prefs"] = {
                "profile.managed_default_content_settings.images": 2,
                "profile.managed_default_content_settings.css": 2
            }
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 20)

            url = "https://www.imdb.com/"
            driver.get(url)
            return driver, wait

        # Research a single movie
        def find_movie_year(movie_name, driver, wait):
            typetext = wait.until(EC.element_to_be_clickable(
                (By.ID, "suggestion-search")))
            typetext.click()
            typetext.send_keys(movie_name.strip())
            typetext.submit()
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            year = soup.find(
                "span", class_="ipc-metadata-list-summary-item__li").text
            try:
                year = int(year)
            except:
                try:
                    year = int(year.split("–")[0])
                except:
                    year = np.nan
            finally:
                return year

        # Research n movies
        def get_n_movie_years(movie_names, dict_output={}):
            driver, wait = open_driver("https://www.imdb.com/")
            for movie_name in movie_names:
                if movie_name not in dict_output:
                    dict_output[movie_name] = find_movie_year(
                        movie_name, driver, wait)
            driver.close()
            return dict_output

        # Parallelize the research by splitting the movies among the workers
        def multiple_windows(movie_list, dict_output={}, workers=5):
            movie_chunks = np.array_split(movie_list, workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                executor.map(get_n_movie_years, [
                    list(chunk) for chunk in movie_chunks], repeat(dict_output))
            return dict_output

        # Get the movie titles
        if not movie_titles:
            movie_titles = self.full["movie_title"].head(10).tolist()

        dict_output = multiple_windows(movie_titles, workers=workers)
        df_output = pd.DataFrame(list(dict_output.items()), columns=[
            "movie_title", "release_year"])

        if save:
            df_output.to_csv(path, index=False)

        return df_output

    # ># Method 6:

    def enrich_with_release_year(self, path: str = "mdo/data/release_years.csv", replace: bool = True) -> pd.DataFrame:
        """Enrich the dataset with the release years of the movies.

        Parameters
        ----------
        path : str, optional
            Path to the release years file, by default "mdo/data/release_years.csv"
        replace : bool, optional
            Replace the full dataset with the enriched one, by default True

        Returns
        -------
        pd.DataFrame
            Enriched dataset
        """

        # Load the release years
        release_years = pd.read_csv(path)

        # Merge the datasets
        enriched_data = self.full.merge(
            release_years, on="movie_title", how="left")

        if replace:
            self.full = enriched_data
            self.pred_columns += ["release_year"]

        return enriched_data

    # ># Method 7:

    def enrich_all(
            self,
            save: bool = False,
            path: str = "mdo/data/enriched_dataset.csv",
            country: bool = True, actors: bool = True, release_year: bool = True,
            country_par: dict = {}, actors_par: dict = {}, release_par: dict = {}) -> None:
        """Enrich the data with all the available methods.

        Parameters
        ----------
        save : bool, optional
            Save the enriched dataset to a file, by default False
        path : str, optional
            Path to save the enriched dataset, by default "mdo/data/enriched_dataset.csv"
        country : bool, optional
            Enrich based on the country where the movie was created, by default True
        actors : bool, optional
            Enrich based on the actors network, by default True
        release_year : bool, optional
            Enrich based on the release year of the movie, by default True
        country_par : dict, optional
            Parameters for the enrich_with_country method, by default {}
        actors_par : dict, optional
            Parameters for the enrich_with_actors method, by default {}
        release_par : dict, optional
            Parameters for the enrich_with_release_year method, by default {}

        Returns
        -------
        pd.DataFrame
            Enriched dataset
        """

        # Add profit to the success metrics
        self.full["profit"] = self.full["gross"] - self.full["budget"]
        self.success_columns = list(self.success_columns) + ["profit"]

        # Enrich the dataset with country and actors
        if country:
            self.enrich_with_country(**country_par)
        if actors:
            self.enrich_with_actors(**actors_par)
        if release_year:
            self.enrich_with_release_year(**release_par)

        # Save the dataset
        if save:
            self.full.to_csv(path, index=False)

        return self.full


class MDO_Statistician(MDO_Researcher):
    """First employee of the Movie Director Office, the statistician is in charge of the data exploration.
    He can provide summary statistics, correlation analysis and ANOVA tests.

    Base Class
    ----------
    MDO_Researcher : MDO_Researcher
        Support the statistician with the data preparation and enrichment.

    Special Methods
    ---------------
    get_dist_financials(plot:bool=True) -> pd.DataFrame
        Give summary statistics about the main financials of movies.

    get_dist_scores(plot:bool=True) -> pd.DataFrame
        See if the IMDB scores are well distributed.

    get_correlation(target:str, predictors:list=[], plot:bool=False) -> pd.DataFrame
        Correlation analysis between the target variable and the chosen predictors.

    get_anova(target:str, predictors:list=[]) -> pd.DataFrame
        ANOVA analysis between the target variable and the chosen predictors.

    show_me_the_world() -> None
        Show an interactive map of the world with movie statistics by country.
    """

    # ># Method 1:

    def get_dist_financials(self, plot: bool = True) -> pd.DataFrame:
        """Give summary statistics about the main financials of movies.

        Parameters
        ----------
        plot : bool, optional
            Plot the distribution, by default True

        Returns
        -------
        pd.DataFrame
            Summary statistics of the budget, gross and profit
        """
        # Describe the financial variables
        financials = ["budget", "gross", "profit"]
        fin_stats = self.full[financials].describe()

        # Plot the distribution of the financial variables
        if plot:
            financials = ["budget", "gross", "profit"]
            colors = ["r", "b", "g"]
            _, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, ax in enumerate(axs):
                sns.histplot(
                    data=self.full, x=financials[i], ax=ax, kde=True,
                    color=colors[i], edgecolor="white")
                for j in ["25%", "50%", "75%"]:
                    ax.axvline(fin_stats.loc[j, financials[i]], color="black")
            plt.suptitle(
                f'Financial distributions of the {self.full.shape[0]} movies')
            plt.show()

        # Return the financials and the best and worst in class
        fin_stats.loc["movie_highest"] = self.full.loc[self.full[financials].idxmax(
        ), "movie_title"].values
        fin_stats.loc["movie_lowest"] = self.full.loc[self.full[financials].idxmin(
        ), "movie_title"].values
        return fin_stats

    # ># Method 2:

    def get_dist_scores(self, plot: bool = True) -> pd.DataFrame:
        """See if the IMDB scores are well distributed.

        Parameters
        ----------
        plot : bool, optional
            Plot the distribution, by default True

        Returns
        -------
        pd.DataFrame
            Summary statistics of the IMDB scores        
        """
        # Plot the distribution of the IMDB scores
        score_stats = self.full["imdb_score"].describe()

        if plot:
            sns.histplot(self.full["imdb_score"], kde=True, edgecolor="white")
            for i in ["25%", "50%", "75%"]:
                plt.axvline(score_stats[i], color="black")
            plt.title(
                f"IMDB scores distribution of the {self.full.shape[0]} movies")
            plt.show()

        return score_stats

    # ># Method 3:

    def get_correlation(self, target: str, predictors: list = [], plot: bool = False) -> pd.DataFrame:
        """Correlation analysis between the target variable and the chosen predictors.

        Parameters
        ----------
        target : str
            Variable to predict
        predictors : list, optional
            List of the predictors to take into account, by default []
        plot : bool, optional
            Plot a graph of the correlations, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with the correlation values
        """

        # Get predictors if not provided and compute the correlations
        if not predictors:
            cond_in_pred = self.full.columns.isin(self.pred_columns)
            cond_not_obj = self.full.dtypes != "object"
            predictors = self.full.columns[cond_in_pred & cond_not_obj]

        corr = pd.DataFrame(columns=[f"correlation_{target}", "p-value"])
        for pred in predictors:
            corr.loc[pred] = pearsonr(self.full[pred], self.full[target])
        corr["significant"] = (corr["p-value"] < 0.05).astype(int)
        corr.sort_values(
            by=f"correlation_{target}", ascending=False, inplace=True)

        if plot:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=corr.index, y=corr[f"correlation_{
                        target}"], hue=corr["significant"], palette=["r", "g"])
            plt.xticks(rotation=45, ha="right")
            plt.title("".join([
                f"Correlation with {target}",
                f" (max: {corr[f"correlation_{target}"].max():.2%},",
                f" min: {corr[f"correlation_{target}"].min():.2%})"]))
            plt.xlabel("")
            plt.show()

        return corr.sort_values(by="p-value", ascending=True)

    # ># Method 4:

    def get_anova(self, target: str, predictors: list = []) -> pd.DataFrame:
        """ANOVA analysis between the target variable and the chosen predictors.

        Parameters
        ----------
        target : str
            Variable to predict
        predictors : list, optional
            List of the predictors to take into account, by default []

        Returns
        -------
        pd.DataFrame
            For each predictor, the F-statistic and p-value of the ANOVA
        """

        # Get predictors if not provided and compute the ANOVA
        if not predictors:
            cond_in_pred = self.full.columns.isin(
                self.pred_columns) & ~self.full.columns.isin(["movie_title"])
            cond_obj = self.full.dtypes == "object"
            predictors = self.full.columns[np.logical_and(
                cond_in_pred, cond_obj)]
        anova = pd.DataFrame()
        for pred in predictors:
            groups = self.full.groupby(pred)[target].apply(list)
            f_stat, p_val = f_oneway(*groups)
            anova.loc[pred, f"F-stat {target}"] = f_stat
            anova.loc[pred, "p-value"] = p_val

        # Return the ANOVA
        return anova.sort_values(by="p-value")

    # ># Method 5:

    def show_me_the_world(self) -> None:
        """Show an interactive map of the world with movie statistics by country.

        Returns
        -------
        None
            Displays the interactive map
        """

        # Aggregating the data by country
        country_view = self.full.groupby("country").agg({
            "movie_title": "count",
            "duration": "mean",
            "language": pd.Series.mode,
            "budget": "mean",
            "gross": "mean",
            "imdb_score": "mean"
        })

        country_view.columns = [
            "Number of movies",
            "Mean duration",
            "Most frequent language",
            "Mean movie budget",
            "Mean box office",
            "Mean IMDB score"
        ]

        # Rounding the values for better readability in the map
        country_view[["Mean duration", "Mean movie budget", "Mean box office"]] = country_view[
            ["Mean duration", "Mean movie budget", "Mean box office"]].apply(lambda x: round(x, 0), axis=1)
        country_view["Mean IMDB score"] = country_view["Mean IMDB score"].apply(
            lambda x: round(x, 2))

        # Adding the country name as a column for easier merging
        country_view["NAME"] = country_view.index

        # Loading the shapefile of the countries boundaries
        countries_shape = gpd.read_file(
            "mdo/data/countries_shape/ne_110m_admin_0_countries.shp")
        countries_shape = countries_shape[[
            "NAME", "POP_EST", "GDP_MD", "geometry"]]
        countries_shape.NAME = countries_shape.NAME.str.replace(
            "United States of America", "USA")
        countries_shape.NAME = countries_shape.NAME.str.replace(
            "United Kingdom", "UK")
        countries_shape.NAME = countries_shape.NAME.str.replace(
            "Czechia", "Czech Republic")

        # Merging the information dataset with the shapefile
        countries_full = countries_shape.merge(country_view, on="NAME")
        countries_full = countries_full.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        countries_full.rename(columns={
            "NAME": "Country",
            "POP_EST": "Estimated population",
            "GDP_MD": "Estimated GDP (Mln$)",
        }, inplace=True)

        # Creating the interactive map
        map_title = "The world of movies"
        title_html = """
            <h3>{map_title}</h3>
            <p>Explore the map and click on countries to see the details</p>
        """.format(map_title=map_title)

        m = folium.Map(width=800, height=300)
        m.get_root().html.add_child(folium.Element(title_html))

        m = countries_full.explore(
            m=m,
            column="Country",
            scheme="naturalbreaks",
            tooltip=[
                "Country",
            ],
            popup=[
                "Country",
                "Estimated population",
                "Estimated GDP (Mln$)",
                "Number of movies",
                "Mean duration",
                "Most frequent language",
                "Mean movie budget",
                "Mean box office",
                "Mean IMDB score"
            ],
            tooltip_kwds=dict(labels=False),
            name="Countries",
            legend=False
        )

        folium.TileLayer("CartoDB positron", show=True).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)

        display(m)


class MDO_Agent(MDO_Researcher):
    """Second employee of the Movie Director Office, the agent knows his way around the movie industry.
    He can provide insights into directors, actors and their relationships.

    Base Class
    ----------
    MDO_Researcher : MDO_Researcher
        Support the agent with the data preparation and enrichment.

    Special Attributes
    ------------------
    network : nx.Graph
        Network of actors

    Special Methods
    ---------------

    get_top_directors(top:int=5, sort_by:str="gross") -> pd.DataFrame
        Get the top directors based on a given criteria and display their information.

    get_info_celebrity(name:str, verbose:bool=True) -> dict
        Fetch information about a celebrity (director or actor) from a trusted colleague.

    create_actors_network(
        save_to:str="mdo/data/actors.pickle",
        load_from:str=None, plot:bool=False) -> nx.Graph
        Create a network of actors based on the movies they played in.

    show_nth_cluster(n:int=1) -> None
        Show the nth component of the actors network.

    show_n_neighbors(actor:str, degree:int=1, plot:bool=False) -> List[str]
        Show the neighbors of an actor in the network up to n connections.
    """

    def __init__(self, **class_arg):
        super().__init__(**class_arg)
        self.network = None

    # ># Method 1:

    def get_top_directors(self, top: int = 5, sort_by: str = "gross") -> pd.DataFrame:
        """Get the top directors based on a given criteria.

        Parameters
        ----------
        top : int, optional
            Number of directors to display, by default 5
        criteria : str, optional
            Criteria to sort the directors, by default "gross"
            Choose from "budget", "gross", "profit", "imdb_score" or "director_facebook_likes"

        Returns
        -------
        pd.DataFrame
            DataFrame with the top directors
        """

        # Define the information to gather for the directors
        criteria = ["budget", "gross", "profit",
                    "imdb_score", "director_facebook_likes"]
        infos = ["director_name", "movie_title", "language", "country"]
        genres = [col for col in self.full.columns if "genre_" in col]

        # Group by director and aggregate the information
        top_directors = self.full[criteria + infos + genres].groupby("director_name").agg({
            **{col: "mean" for col in criteria},
            "movie_title": "count", "language": pd.Series.mode, "country": pd.Series.mode,
            **{col: "sum" for col in genres}
        }).sort_values(by=sort_by, ascending=False).head(top)
        top_directors.columns = [
            "Average budget", "Average gross", "Average profit", "Average imdb score", "Average director facebook likes",
            "Number of movies", "Most common language", "Most common country", *genres
        ]

        # Return the top directors
        return top_directors

    # ># Method 2:

    def get_info_celebrity(self, name: str, verbose: bool = True) -> dict:
        """Get information about a celebrity.

        Parameters
        ----------
        name : str
            Name of the celebrity (director or actor)
        verbose : bool, optional
            Display the information, by default True

        Returns
        -------
        dict
            Information about the celebrity
        """

        # Create the API URL
        api_url = 'https://api.api-ninjas.com/v1/celebrity?name={}'.format(
            name)

        # Use environment variable to keep the API KEY private when shared
        response = requests.get(
            api_url, headers={'X-Api-Key': os.environ.get("NINJA_API_KEY", None)})

        # Catch eventual failed requests
        if response.status_code == requests.codes.ok:
            details = (response.json()[0])
            if verbose:
                print(f"Here is all I know about {name}:")
                for key, value in details.items():
                    print(f"  - {key}: {value}")
            return details
        else:
            print(f"I'm sorry, I don't know much about {name}.")
            return {}

    # ># Method 3:

    def create_actors_network(
            self,
            save_to: str = "mdo/data/actors.pickle",
            load_from: str = None,
            plot: bool = False) -> nx.Graph:
        """Create a network of actors based on the movies they played in.

        Parameters
        ----------
        save_to : str, optional
            Path to save the network, by default "mdo/data/actors.pickle"
        load_from : str, optional
            Path to load the network from, by default None
        plot : bool, optional
            Plot the network, by default False

        Returns
        -------
        nx.Graph
            Network of actors
        """

        if save_to is not None and load_from is None:
            # Reformat existing data to have the movie-actor pairs
            actor1 = self.full[["movie_title", "actor_1_name"]].rename(
                columns={"actor_1_name": "actor"})
            actor2 = self.full[["movie_title", "actor_2_name"]].rename(
                columns={"actor_2_name": "actor"})
            actor3 = self.full[["movie_title", "actor_3_name"]].rename(
                columns={"actor_3_name": "actor"})
            actors = pd.concat([actor1, actor2, actor3], ignore_index=True)
            actors["movie_title"] = actors["movie_title"].apply(
                lambda x: x.strip())

            # Group the movies by actors
            graph_data = actors.groupby(
                "actor")["movie_title"].apply(list).reset_index()

            # Create the network
            self.network = nx.Graph()

            # For each actor, add edges with other actors they played with
            for i, (actor, movies) in graph_data.iterrows():
                self.network.add_node(actor)
                print(
                    f"Network creation: {(i + 1) / len(graph_data):.2%}", end="\r")
                for _, (actor2, movies2) in graph_data.iloc[i + 1:].iterrows():
                    common_movies = set(movies).intersection(set(movies2))
                    if len(common_movies) > 0:
                        self.network.add_edge(
                            actor, actor2, weight=len(common_movies))

            # Save the network
            pickle.dump(self.network, open(save_to, "wb"))

        elif save_to is None and load_from is not None:
            self.network = pickle.load(open(load_from, "rb"))

        else:
            raise ValueError(
                "You need to choose either to save or to load the network")

        # Plot the network
        if plot:
            nx.draw(self.network, with_labels=False, node_size=2)
            plt.show()

        return self.network

    # ># Method 4:

    def show_nth_cluster(self, n: int = 1) -> None:
        """Show the nth component of the actors network.

        Parameters
        ----------
        n : int, optional
            Number of the component to display from largest, by default 1

        Returns
        -------
        None
            Display the network component
        """

        # Get the components and order them by size
        components = sorted(nx.connected_components(
            self.network), key=len, reverse=True)
        chosen_component = list(components[n - 1])

        # Plot the chosen component
        subgraph = self.network.subgraph(chosen_component)
        nx.draw(subgraph, with_labels=True, node_size=10,
                font_size=10, edge_color="lightgrey")
        plt.title(
            f"Component {n} of the actors network ({len(chosen_component)} actors)")
        plt.show()

    # ># Method 5:

    def show_n_neighbors(self, actor: str, degree: int = 1, plot: bool = False) -> list:
        """Show the neighbors of an actor in the network up to n connections.

        Parameters
        ----------
        actor : str
            Name of the actor
        degree : int, optional
            Number of neighbors to display, by default 1
        plot : bool, optional
            Plot the network, by default False

        Returns
        -------
        List[str]
            List of the neighbors

        Raises
        ------
        KeyError
            If the actor is not found in the network
        """

        # Get the neighbors of the actor
        sub_nodes = [actor]
        try:
            sub_nodes = list(nx.single_source_shortest_path(
                self.network, actor, cutoff=degree).keys())
        except KeyError:
            raise KeyError(f"{actor} is not in the network")

        # Show the subgraph of the neighbors
        if plot:
            subgraph = self.network.subgraph(sub_nodes)
            color_map = [
                "lightgreen" if node == actor else "lightblue" for node in subgraph.nodes()]
            size_map = [100 if node ==
                        actor else 10 for node in subgraph.nodes()]
            nx.draw(
                subgraph, with_labels=True, node_size=size_map, node_color=color_map,
                font_size=10, edge_color="lightgrey")
            plt.title(
                f"Neighbors of {actor} up to degree {degree} (n={len(sub_nodes) - 1})")
            plt.show()

        sub_nodes.remove(actor)
        sub_nodes.sort()
        return sub_nodes


class MDO_Forecaster(MDO_Researcher):
    """Last employee of the Movie Director Office, the forecaster is in charge of predicting the success of
    a movie before it is released. He can build models and evaluate their performance.

    Base Class
    ----------
    MDO_Researcher : MDO_Researcher
        Support the forecaster with the data preparation and enrichment.

    Special Methods
    ---------------
    prepare_X_y(
        exclude:list=["movie_title", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres"],
        subset:list=[], target:str="gross", verbose:bool=True) -> None
        Prepare the predictors for the model.

    initiate_models(
        models:list=["DummyRegressor", "LinearRegression", "Lasso", "SVR", "KNeighborsRegressor",
        "RandomForestRegressor", "XGBRegressor", "MLPRegressor"],
        standard_scaling:bool=True) -> dict
        Initiate the models to be used for prediction.

    evaluate_standard_models(
        cv:int=5, logs:str="logs.csv", rank:str="mse", prefix:str="", verbose:bool=True) -> pd.DataFrame
        Evaluate the performance of the standard models.

    evaluate_single_model(
        estimator:model, cv:int=5, logs:str="logs.csv", rank:str="mse",
        name_log:str='', plot_pred:bool=False, scatter_plot:bool=False,
        verbose:bool=True) -> pd.DataFrame
        Evaluate the performance of a single model.

    get_OLS_significance(
        backward_elimination:bool=True, alpha:float=0.05, log:bool=True,
        log_path:str="OLS_logs.csv", log_name:str="OLS",
        explainable:bool=False, verbose:bool=True) -> dict

    classify_profitable(
        model:model, X:pd.DataFrame=None, log:bool=True,
        log_path:str="profitable_classifier.csv", log_name:str="Classifier") -> pd.DataFrame
    """

    def __init__(self, pre: str = "mdo/data/pre_release.csv", after: str = "mdo/data/after_release.csv") -> None:
        super().__init__(pre, after)
        self.models = {}
        self.X = None
        self.y = None
        self.logs = None

    # ># Method 1:

    def prepare_X_y(
            self,
            exclude: list = [
                "movie_title", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres"
            ],
            subset: list = [],
            target: str = "gross",
            verbose: bool = True) -> None:
        """Prepare the predictors for the model.

        Parameters
        ----------
        exclude : list, optional
            Columns to exclude from the predictors, by default [
                "movie_title", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres"
            ]
        subset : list, optional
            Subset of the columns to keep, by default []
        target : str, optional
            Target variable for the model, by default "gross"
        verbose : bool, optional
            Display execution details, by default True
        """

        # Define the predictors and the target
        self.y = self.full[target]

        # Keep only the predictors
        self.X = self.full[self.pred_columns].drop(columns=exclude)

        if subset != []:
            self.X = self.X[subset]

        if verbose:
            print(f"{self.X.shape[1]} initial predictors")

        # Dummy encode the categorical variables
        self.X = pd.get_dummies(self.X, drop_first=True)
        if verbose:
            print(f"{self.X.shape[1]} predictors after encoding")

        # Convert boolean columns to integers for compatibility with some models
        for col in self.X.columns:
            if self.X[col].dtype == bool:
                self.X[col] = self.X[col].astype(int)

        # Filter out missing values
        if verbose:
            print(f"{self.X.shape[0]} rows before filtering missing values")
        self.X.dropna(inplace=True)
        self.y = self.y[self.X.index]
        if verbose:
            print(f"{self.X.shape[0]} rows after filtering missing values")

    # ># Method 2:

    def initiate_models(self, models: list = [
            "DummyRegressor", "LinearRegression", "Lasso", "SVR", "KNeighborsRegressor",
            "RandomForestRegressor", "XGBRegressor", "MLPRegressor"],
            standard_scaling: bool = True) -> dict:
        """Initiate the models to be used for prediction.

        Parameters
        ----------
        models : list, optional
            List of models to initiate, by default [
                "DummyRegressor", "LinearRegression", "Lasso", "SVR", "KNeighborsRegressor",
                "RandomForestRegressor", "XGBRegressor", "MLPRegressor"
            ]
        standard_scaling : bool, optional
            Standardize the predictors, by default True

        Returns
        -------
        dict
            Dictionary of initiated models
        """

        # Initiate the models
        if "DummyRegressor" in models:
            self.models["DummyRegressor"] = make_pipeline(
                StandardScaler() if standard_scaling else None, DummyRegressor())
        if "LinearRegression" in models:
            self.models["LinearRegression"] = make_pipeline(
                StandardScaler() if standard_scaling else None, LinearRegression())
        if "Lasso" in models:
            self.models["Lasso"] = make_pipeline(
                StandardScaler() if standard_scaling else None, Lasso())
        if "KNeighborsRegressor" in models:
            self.models["KNeighborsRegressor"] = make_pipeline(
                StandardScaler() if standard_scaling else None, KNeighborsRegressor())
        if "SVR" in models:
            self.models["SVR"] = make_pipeline(
                StandardScaler() if standard_scaling else None, SVR(kernel="rbf"))
        if "RandomForestRegressor" in models:
            self.models["RandomForestRegressor"] = make_pipeline(
                StandardScaler() if standard_scaling else None, RandomForestRegressor())
        if "XGBRegressor" in models:
            self.models["XGBRegressor"] = make_pipeline(
                StandardScaler() if standard_scaling else None, XGBRegressor())
        if "MLPRegressor" in models:
            self.models["MLPRegressor"] = make_pipeline(
                StandardScaler() if standard_scaling else None, MLPRegressor())

        return self.models

    # ># Method 3:

    def evaluate_standard_models(
            self, cv: int = 5, logs: str = "logs.csv", rank: str = "mse",
            prefix: str = "", verbose: bool = True) -> pd.DataFrame:
        """Evaluate the performance of the standard models.

        Parameters
        ----------
        cv : int, optional
            Number of cross-validation folds, by default 5
        logs : str, optional
            Path to save the logs, by default "logs.csv"
        rank : str, optional
            Metric to rank the models, by default "mse"
        prefix : str, optional
            Prefix to add to the model names, by default ""
        verbose : bool, optional
            Display the evaluation output_dict, by default True

        Returns
        -------
        pd.DataFrame
            Logs of the model evaluation
        """

        # Check if the logs csv exists and create pandas table if not
        if not os.path.exists(logs):
            self.logs = pd.DataFrame(
                columns=["model", "mse", "mape", "r2", "timestamp"])
            self.logs.to_csv(logs, index=False)

        # Read the logs file into the dataframe
        self.logs = pd.read_csv(logs)

        # Set up the scorer
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_mape": "neg_mean_absolute_percentage_error",
            "r2": "r2"
        }

        for model_name, model in self.models.items():
            if verbose:
                print(f"Evaluating {model_name}")

            scores = cross_validate(
                model, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

            # Add the results to the logs
            if prefix + model_name not in self.logs["model"].values:
                self.logs.loc[len(self.logs)] = {
                    "model": prefix + model_name,
                    "mse": -scores["test_neg_mse"].mean(),
                    "mape": -scores["test_neg_mape"].mean(),
                    "r2": scores["test_r2"].mean(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                self.logs.sort_values(by=rank, inplace=True)

                # Save the logs
                self.logs.to_csv(logs, index=False)

        return self.logs

    # ># Method 4:

    def evaluate_single_model(
            self, estimator, cv: int = 5, logs: str = "logs.csv", rank: str = "mse",
            name_log: str = "", plot_pred: bool = False, scatter_plot: bool = False,
            verbose: bool = True) -> pd.DataFrame:
        """Evaluate the performance of a chosen estimator.

        Parameters
        ----------
        estimator : model
            Model to evaluate
        cv : int, optional
            Number of cross-validation folds, by default 5
        logs : str, optional
            Path to save the logs, by default "logs.csv"
        rank : str, optional
            Metric to rank the models, by default "mse"
        name_log : str, optional
            Name to register the model in the logs, by default ""
        plot_pred : bool, optional
            Plot the predictions vs the true values, by default False
        scatter_plot : bool, optional
            Plot the scatter plot of the predictions vs the true values, by default False
        verbose : bool, optional
            Display the evaluation results, by default True

        Returns
        -------
        pd.DataFrame
            Logs of the model evaluation
        """

        # Check if the logs csv exists and create pandas table if not
        if not os.path.exists(logs):
            self.logs = pd.DataFrame(
                columns=["model", "mse", "mape", "r2", "timestamp"])
            self.logs.to_csv(logs, index=False)

        # Read the logs file into the dataframe
        self.logs = pd.read_csv(logs)

        # Set up the scorer
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_mape": "neg_mean_absolute_percentage_error",
            "r2": "r2"
        }

        scores = cross_validate(
            estimator, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

        model_evaluation = {
            "model": name_log,
            "mse": -scores["test_neg_mse"].mean(),
            "mape": -scores["test_neg_mape"].mean(),
            "r2": scores["test_r2"].mean(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        if verbose:
            print(f"{model_evaluation['timestamp']}: {name_log}")
            for metric in ["mse", "mape", "r2"]:
                print(f"  - {metric}: {model_evaluation[metric]:.4f}")

        # Add the results to the logs
        if name_log not in self.logs["model"].values:
            self.logs.loc[len(self.logs)] = {
                "model": name_log,
                "mse": -scores["test_neg_mse"].mean(),
                "mape": -scores["test_neg_mape"].mean(),
                "r2": scores["test_r2"].mean(),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            self.logs.sort_values(by=rank, inplace=True)

            # Save the logs
            self.logs.to_csv(logs, index=False)

        # Plot the predictions
        if plot_pred or scatter_plot:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2)
            estimator.fit(X_train, y_train)
            y_train = y_train.sort_values()
            X_train = X_train.loc[y_train.index]
            y_test = y_test.sort_values()
            X_test = X_test.loc[y_test.index]
            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)

            for dataset in [y_train, y_test, X_train, X_test]:
                dataset.reset_index(drop=True, inplace=True)

            if plot_pred:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                ax1.plot(y_train)
                ax1.plot(y_train_pred)
                ax1.set_title("Train set")
                ax1.legend(["True", "Predicted"])
                ax2.plot(y_test)
                ax2.plot(y_test_pred)
                ax2.set_title("Test set")
                ax2.legend(["True", "Predicted"])
                fig.suptitle(f"Predictions of {y_train.name} with {name_log}")
                plt.show()

            if scatter_plot:
                max_train_corr = pd.concat([
                    y_train, X_train], axis=1).corr()[y_train.name].abs().sort_values(ascending=False).iloc[1]
                max_test_corr = pd.concat([
                    y_test, X_test], axis=1).corr()[y_test.name].abs().sort_values(ascending=False).iloc[1]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                ctest_train = pearsonr(y_train, y_train_pred)
                sns.regplot(x=y_train, y=y_train_pred, ax=ax1, ci=None)
                ax1.set_title("".join([
                    f"Train set: correlation of {
                        ctest_train[0]:.2%} (pval: {ctest_train[1]:.2%})",
                    f"\n Most correlated predictor: {max_train_corr:.2%} (abs)"]))
                ax1.set_xlabel("True values")
                ax1.set_ylabel("Predicted values")

                ctest_test = pearsonr(y_test, y_test_pred)
                sns.regplot(x=y_test, y=y_test_pred, ax=ax2, ci=None)
                ax2.set_title("".join([
                    f"Test set: correlation of {
                        ctest_test[0]:.2%} (pval: {ctest_test[1]:.2%})",
                    f"\n Most correlated predictor: {max_test_corr:.2%} (abs)"]))
                ax2.set_xlabel("True values")
                ax2.set_ylabel("Predicted values")

                fig.suptitle(f"Predictions of {y_train.name} with {name_log}")

                plt.tight_layout()
                plt.show()

        return self.logs

    # ># Method 5:

    def get_OLS_significance(
            self, backward_elimination: bool = False,
            alpha: float = 0.05, log: bool = True,
            log_path: str = "OLS_logs.csv", log_name: str = "OLS",
            explainable: bool = False, verbose: bool = True) -> dict:
        """Perform an OLS regression and check the significance of the predictors.

        Parameters
        ----------
        backward_elimination : bool, optional
            Perform backward elimination, by default False
        alpha : float, optional
            Significance level, by default 0.05
        log : bool, optional
            Log the results, by default True
        log_path : str, optional
            Path to the log file, by default "OLS_logs.csv"
        log_name : str, optional
            Name to register the model in the logs, by default "OLS"
        explainable : bool, optional
            Make the coefficients explainable with relative CI and permutation importance, by default False
        verbose : bool, optional
            Display the backward elimination details, by default True

        Returns
        -------
        dict
            Results of the OLS regression {Global R2, Coefficients, Logs}
        """

        # Prepare the model
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()

        # Perform backward elimination
        if backward_elimination:
            while True:
                pvalues = model.pvalues
                if pvalues.max() > alpha:
                    to_drop = pvalues.idxmax()
                    if verbose:
                        print(
                            f"Removing '{
                                to_drop}' with p-value {pvalues.max():.2f}",
                            f"\n   -> R²: {model.rsquared:.2f} for {X.shape[1] - 1} predictors")
                    X.drop(columns=to_drop, inplace=True)
                    model = sm.OLS(self.y, X).fit()
                else:
                    break

        output_dict = {
            "Global R2": model.rsquared, "Coefficients": model.summary2().tables[1]}

        # Make the coefficients more explainable

        if explainable:

            coef = output_dict["Coefficients"]
            coef["LB/Coef"] = coef["[0.025"] / coef["Coef."]
            coef["UB/Coef"] = coef["0.975]"] / coef["Coef."]
            coef["Relative Range"] = coef["UB/Coef"] - coef["LB/Coef"]
            if "const" in coef.index:
                coef.drop(index="const", inplace=True)
            coef.sort_values(by="P>|t|", ascending=True, inplace=True)

            def OLS_permutation_importance(X, y, crossval=5, seed=0):
                """Compute the permutation importance of the predictors.

                Parameters
                ----------
                X : pd.DataFrame
                    Predictors
                y : pd.Series
                    Target variable
                crossval : int, optional
                    Number of cross-validation folds, by default 5
                seed : int
                    Random seed

                Returns
                -------
                pd.Series
                    Permutation importance of the predictors
                """

                X = sm.add_constant(X)
                base_r2 = sm.OLS(y, X).fit().rsquared

                feature_importance = pd.DataFrame(
                    columns=["Feature", "R2 impact"])
                np.random.seed(seed)
                for col in X.columns[1:]:
                    importances = []
                    for _ in range(crossval):
                        X_perm = X.copy()
                        X_perm.loc[:, col] = np.random.permutation(X_perm[col])
                        perm_r2 = sm.OLS(y, X_perm).fit().rsquared
                        importances.append(base_r2 - perm_r2)
                    feature_importance.loc[len(feature_importance)] = {
                        "Feature": col, "R2 impact": np.mean(importances)}
                return feature_importance

            permutation_importance = OLS_permutation_importance(X, self.y)

            coef = coef.merge(
                permutation_importance, left_index=True, right_on="Feature", how="left")
            coef.set_index("Feature", inplace=True)

            output_dict["Coefficients"] = coef

            if verbose:
                print(f"The model can explain {
                    output_dict["Global R2"]:.2%} of the variance in the gross revenue")
                print(">>> Significant predictors:")
                display(output_dict["Coefficients"])
                sns.barplot(x="Feature", y="R2 impact",
                            data=output_dict["Coefficients"].head(10))
                plt.title("Permutation importance of the top 10 predictors")
                plt.xticks(rotation=45, ha="right")
                plt.show()

        # Create the log file if it does not exist
        if log:
            if not os.path.exists(log_path):
                pd.DataFrame(
                    columns=[
                        "name", "global_r2",
                        "n_coefficients", "timestamp"]
                ).to_csv(log_path, index=False)

            logs_df = pd.read_csv(log_path)

            if log_name not in logs_df["name"].values:
                logs_df.loc[len(logs_df)] = {
                    "name": log_name,
                    "global_r2": model.rsquared,
                    "n_coefficients": X.shape[1],
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                logs_df.sort_values(
                    by="global_r2", ascending=False, inplace=True)
                logs_df.to_csv(log_path, index=False)

                if verbose:
                    print(f"Model '{log_name}' saved in the logs")
                    display(logs_df.head(5))

            else:
                if verbose:
                    print(f"Model '{log_name}' saved in the logs")
                    display(logs_df.head(5))

            output_dict["Logs"] = logs_df

        return output_dict

    # ># Method 6:

    def classify_profitable(
            self, model, X: pd.DataFrame = None,
            log: bool = True, log_path: str = "profit_classifier.csv",
            log_name: str = "Classifier") -> pd.DataFrame:
        """Classify the movies as profitable or not.

        Parameters
        ----------
        model : model
            Classifier model
        X : pd.DataFrame, optional
            Predictors, by default None
        log : bool, optional
            Log the results, by default True
        log_path : str, optional
            Path to the log file, by default "profit_classifier.csv"
        log_name : str, optional
            Name to register the model in the logs, by default "Classifier"

        Returns
        -------
        pd.DataFrame
            Logs of the model evaluation ordered by F1 score
        """

        if self.y.name != "profit" and self.y.name != "ROI":
            raise ValueError(
                "The target variable should be the profit or ROI of the movie")

        y = (self.y > 0).astype(int)

        if X is None:
            X = self.X

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }

        scores = cross_validate(
            model, X, y, cv=5, scoring=scoring, n_jobs=-1, return_train_score=False)

        if log:
            if not os.path.exists(log_path):
                pd.DataFrame(
                    columns=[
                        "name", "accuracy", "precision",
                        "recall", "f1", "timestamp"]
                ).to_csv(log_path, index=False)

            logs_df = pd.read_csv(log_path)

            if log_name not in logs_df["name"].values:
                logs_df.loc[len(logs_df)] = {
                    "name": log_name,
                    "accuracy": scores["test_accuracy"].mean(),
                    "precision": scores["test_precision"].mean(),
                    "recall": scores["test_recall"].mean(),
                    "f1": scores["test_f1"].mean(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                logs_df.sort_values(by="f1", ascending=False, inplace=True)
                logs_df.to_csv(log_path, index=False)

        return logs_df
