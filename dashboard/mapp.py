import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from io import BytesIO
import folium
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(page_title="Pollution Dashboard", layout="wide")
st.title("Pollution Comparison & Tree Recommendations")

mode= st.sidebar.radio("Select Mode:",["Single City Analysis", "Multiple city comparison"])


mapping= {
    "PM2.5": ["Ficus benghalensis (Banyan)", "Ficus religiosa (Peepal)", "Mangifera indica (Mango)", "Azadirachta indica (Neem)", "Delonix regia (Gulmohar)"],
    "PM10": ["Ficus benghalensis (Banyan)", "Ficus religiosa (Peepal)", "Mangifera indica (Mango)", "Azadirachta indica (Neem)", "Delonix regia (Gulmohar)"],
    "NO2": ["Ficus spp. (Banyan/Peepal)", "Azadirachta indica (Neem)", "Pongamia pinnata (Karanja)", "Evergreen broadleaves"],
    "SO2": ["Ficus spp. (Banyan/Peepal)", "Azadirachta indica (Neem)", "Pongamia pinnata (Karanja)", "Evergreen broadleaves"],
    "Ozone": ["Ficus spp.(Banyan/Peepal)", "Mangifera indica(Mango)", "Syzygium cumini (Jamun)"],
    "CO": ["Ficus (Banyan/Peepal)", "Mango", "Neem"],
    "NH3": ["Azadirachta indica (Neem)", "Moringa oleifera (Moringa)", "Ocimum sanctum (Tulsi)", "Pongamia pinnata (Pongamia)", "Vetiver grass"]
}

                             # -------------------- Wind Patterns (Logic) --------------------
SEASON_WIND = {
    "Winter": "NE → SW",
    "Pre-monsoon": "SW → NE",
    "Monsoon": "SW → NE (strong)",
    "Post-monsoon": "E → W"
}

def month_to_season(month):
    if month in [12,1,2]: return "Winter"
    elif month in [3,4,5]: return "Pre-monsoon"
    elif month in [6,7,8,9]: return "Monsoon"
    else: return "Post-monsoon"



   
                            ##########################-----Multiple-city-comparison-------
if mode =="Multiple city comparison":
    st.header("Comparison Analysis")


    @st.cache_data
    def load_and_prepare(delhi_path="delhi_mly.csv", bbsr_path="bbsr_yearly.csv"):
        d = pd.read_csv(delhi_path)
        b = pd.read_csv(bbsr_path)

        d.columns = d.columns.str.strip().str.replace(r'\s+', '', regex=True)
        b.columns = b.columns.str.strip().str.replace(r'\s+', '', regex=True)

        for df in (d, b):
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)

        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']
        for col in pollutants:
            if col not in d.columns: d[col] = np.nan
            if col not in b.columns: b[col] = np.nan

        d['date'] = pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str).str.zfill(2) + '-01')
        b['date'] = pd.to_datetime(b['year'].astype(str) + '-' + b['month'].astype(str).str.zfill(2) + '-01')

        df = pd.concat([d, b], ignore_index=True, sort=False)
        df = df.sort_values(['city', 'date']).reset_index(drop=True)

        for col in pollutants:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['city'] = df['city'].str.strip().str.title()
        return df


    df = load_and_prepare()
    if df is None:
        st.stop()

                            # -------------------- Sidebar Controls --------------------
    st.sidebar.header("Filter")
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    years = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year,
                            value=(min_year, max_year), step=1)

    possible_pollutants = [c for c in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3'] if c in df.columns]
    default_selection = [p for p in ['PM2.5', 'PM10', 'NO2'] if p in possible_pollutants]
    pollutantss = st.sidebar.multiselect("Select pollutants to compare", options=possible_pollutants,
                                        default=default_selection)

    chart_type = st.sidebar.selectbox("Chart type", [
        "None",
        "Line (time series)",
        "Monthly average(1-12)",
        "Boxplot (distribution)",
        "Heatmap (seasonal patterns)"
    ])

    cities = df['city'].unique().tolist()


    st.sidebar.markdown("---")
    pollutant_options= ["None"] + possible_pollutants
    single_pollutant= st.sidebar.selectbox(
        "Select a pollutant for recommended actions",
        options=pollutant_options,
        index=0 
    )


    selected_cities = st.sidebar.multiselect("Cities to display", options=cities, default=cities)

                            
                            # -------------------- Filtering the years --------------------
    start_year, end_year = years
    df_filt = df[(df['year'] >= start_year) & (df['year'] <= end_year) & (df['city'].isin(selected_cities))].copy()

    if df_filt.empty:
        st.warning("No data available for selected filter")
        st.stop()

    st.write("### Data summary")
    st.write(f"Selected years: {start_year}-{end_year}")
    st.write(df_filt[['city', 'year', 'month']].drop_duplicates().groupby('city').size().rename('rows per city'))

                                
                                
                                
                                
                                # -------------------- Plot Functions --------------------
    def plot_time_series(df_sel, pollutant):
        fig, ax = plt.subplots(figsize=(10, 4))
        for city in selected_cities:
            city_df = df_sel[df_sel['city'] == city].sort_values('date')
            if pollutant in city_df.columns:
                ax.plot(city_df['date'], city_df[pollutant], marker='o', label=city)
        ax.set_title(f"{pollutant} - Time Series ({start_year}-{end_year})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Concentration")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)


    def plot_monthly_average(df_sel, pollutant):
        m = df_sel.groupby(['city', 'month'])[pollutant].mean().unstack(level=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        months = range(1, 13)
        for city in selected_cities:
            if city in m.columns:
                ax.plot(months, m[city].reindex(months).values, marker='o', label=city)
        ax.set_xticks(months)
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean concentration")
        ax.set_title(f"{pollutant} — Monthly average (averaged across selected years)")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)


    def plot_boxplot(df_sel, pollutants_list):
        if not pollutants_list:
            st.warning("No pollutants selected.")
            return
        for p in pollutants_list:
            fig, ax = plt.subplots(figsize=(8, 5))
            df_sel.boxplot(column=p, by="city", ax=ax)
            ax.set_title(f"Distribution of {p} by City ({start_year}-{end_year})")
            ax.set_ylabel("Concentration")
            plt.suptitle("")
            st.pyplot(fig)
            plt.close(fig)


    def plot_heatmap(df_sel, city):
        df_city = df_sel[df_sel['city'] == city]
        if df_city.empty:
            st.warning(f"No data for {city}")
            return
        pivot = df_city.groupby("month")[possible_pollutants].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot.T, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
        ax.set_title(f"Monthly Pollution Pattern — {city} ({start_year}-{end_year})")
        st.pyplot(fig)
        plt.close(fig)


                    # -------------------- Choice of plots --------------------
    st.header("Dynamic Visual Comparison")

    if not pollutantss:
        st.info("Choose at least one pollutant from the sidebar to plot")
    else:
        if chart_type == "None":
            st.info("No plot selected.")

        elif chart_type == "Line (time series)":
            st.subheader("Time series (per pollutant)")
            for p in pollutantss:
                if p in df_filt.columns:
                    plot_time_series(df_filt, p)

        elif chart_type == "Monthly average(1-12)":
            st.subheader("Monthly average profile (1-12)")
            for p in pollutantss:
                if p in df_filt.columns:
                    plot_monthly_average(df_filt, p)

        elif chart_type == "Boxplot (distribution)":
            st.subheader("Distribution comparison (Boxplots)")
            plot_boxplot(df_filt, pollutantss)

        elif chart_type == "Heatmap (seasonal patterns)":
            st.subheader("Heatmap of seasonal pollution patterns")
            for city in selected_cities:
                plot_heatmap(df_filt, city)

                        
                        
                        
                        
                                # -------------------- Dominant Pollutant & Tree Recs  mapping--------------------
    st.subheader("Dominant pollutant & Recommended Trees")

    summary_rows = []
    pol_for_summary = [p for p in pollutantss if p in df_filt.columns]

    for city in selected_cities:
        city_means = df_filt[df_filt['city'] == city][pol_for_summary].mean().to_dict()
        if len(city_means) == 0:
            st.write(f"No pollutant means for {city}.")
            continue

        dominant = max(city_means, key=lambda k: (city_means[k] if pd.notna(city_means[k]) else -np.inf))
        domi_val = city_means.get(dominant, np.nan)
        rec_trees = mapping.get(dominant, ["Local species"])
        st.markdown(f"**{city}** - Dominant pollutant: {dominant} (mean = {domi_val:.2f})")
        st.write("Recommended trees:", ",".join(rec_trees))

                                #..........Recommend actions as per dominant pollutant:

    if dominant in ["PM2.5","PM10"]:
            st.write("""
            - Action: \n\n1.Plant Banyan, Peepal, Neem, Mango, Jamun in urban roadsides, markets, schools, and residential belts.
                    \n2.Plant Tulsi hedges near households and temples — cultural + air purification role.
                    \n3.Use layered planting: tall canopy trees + shrubs (like Hibiscus, Bougainvillea) near roads to trap suspended particles.
                    \n4.Plant avenue rows along traffic-heavy corridors (2–3 rows of dense foliage trees).
                    
            - Why it works: Broad leaves, rough surfaces trap dust/soot; multiple canopy layers reduce PM concentration.""")

    elif dominant == ["NO2","SO2"]:
            st.write("""
                    - Action: 1.Plant Neem, Pongamia (Karanja), Peepal, Banyan around industries, power plants, and highways.
                            2.Create buffer greenbelts: at least 30–100 m wide rows of pollutant-tolerant species around emission zones (as CPCB recommends).
                            3.Combine with evergreens like Ashoka (Polyalthia longifolia) to form year-round barriers.
                    - Why it works: High stomatal density trees absorb SO₂ & NO₂; dense canopies dilute and filter gases.
                    """)
    elif dominant == "Ozone":
            st.write("""
                    - Action: 1.Avoid Eucalyptus monocultures (they emit isoprenes → increase O₃).
                            2.Plant Ficus, Jamun, Mango, Neem in city cores.
                            3.Use cooling green corridors to reduce precursor buildup (O₃ rises in high heat + high NOx zones).
                    - Why it works: Shade trees reduce temperature, preventing ozone formation hotspots; low-VOC trees minimize precursors.
                    """)        
    else:
            st.write("- Action: Plant dense trees and wear masks in outdoors.")



                                # -------------------- Recommended Actions (User-selected pollutant) --------------------
    st.subheader("Recommended Actions (user-selected pollutant)")

    if single_pollutant == "None":
        st.info("No pollutant selected. Please choose one in the sidebar to view recommended actions.")
    else:
        st.write(f"**Selected pollutant:** {single_pollutant}")
        rec_trees_selected = mapping.get(single_pollutant, ["Local species"])
        st.write("Recommended trees:", ", ".join(rec_trees_selected))

        if single_pollutant in ["PM2.5", "PM10"]:
            st.write("""
            - **Action:**  
            1. Plant Banyan, Peepal, Neem, Mango, Jamun in urban roadsides, markets, schools, and residential belts.  
            2. Plant Tulsi hedges near households and temples.  
            3. Use layered planting: tall canopy trees + shrubs near roads to trap suspended particles.  
            4. Plant avenue rows along traffic-heavy corridors (2–3 rows of dense foliage trees).  

            - **Why it works:** Broad leaves, rough surfaces trap dust/soot; multiple canopy layers reduce PM concentration.
            """)

        elif single_pollutant in ["NO2", "SO2"]:
            st.write("""
            - **Action:**  
            1. Plant Neem, Pongamia (Karanja), Peepal, Banyan around industries, power plants, and highways.  
            2. Create buffer greenbelts (30–100 m wide) around emission zones.  
            3. Combine with evergreens like Ashoka to form year-round barriers.  

            - **Why it works:** High stomatal density trees absorb SO₂ & NO₂; dense canopies dilute and filter gases.
            """)

        elif single_pollutant == "Ozone":
            st.write("""
            - **Action:**  
            1. Avoid Eucalyptus monocultures.  
            2. Plant Ficus, Jamun, Mango, Neem in city cores.  
            3. Use cooling green corridors to reduce precursor buildup.  

            - **Why it works:** Shade trees reduce local temperature; low-VOC trees minimize precursors.
            """)

        elif single_pollutant == "CO":
            st.write("""
            - **Action:**  
            1. Plant dense roadside vegetation (Banyan, Peepal, Mango, Neem).  
            2. Create micro-forests in residential belts.  

            - **Why it works:** Trees with high leaf surface area help dilute localized vehicle exhaust.
            """)

        elif single_pollutant == "NH3":
            st.write("""
            - **Action:**  
            1. Plant Neem, Moringa, Tulsi, Pongamia, and Vetiver near agricultural zones.  
            2. Use bio-shields around fertilizer-intensive areas.  

            - **Why it works:** These plants absorb ammonia and help manage nitrogen cycles.
            """)

        else:
            st.write("- Action: Plant dense trees and recommended greening; wear masks outdoors when needed.")



    summary_rows.append({
        "City":city,
        "Dominant Pollutant": dominant,
        "Dominant mean": domi_val,
        "Recommended trees":";".join(rec_trees)
    })




                                                # Download Summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        buf = BytesIO()
        summary_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download Summary", data=buf,
                        file_name="pollutant_summary.csv", mime="text/csv")

    st.write("---")
    st.caption("This app uses visual-based dominance. For production-grade systems, we'd add uncertainty estimates, more contextual features, and ML analysis.")



                           ##################SINGLE MODE ANALYSIS#################3



elif mode== "Single City Analysis":
    st.header("Single City Analysis")


    city_datasets ={
        "None":"",
        "Bhubaneswar":"bbsr_en.csv"
    }

    selected_city = st.sidebar.selectbox("Select a city",options= list(city_datasets.keys()),index=0)

    if selected_city == "None":
        st.info("Please select a city to view analysis.")
        st.stop()


    @st.cache_data
    def load_city(path,city_name):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df
    
    city_df = load_city(city_datasets[selected_city],selected_city)


    ##sidebar filters

    st.sidebar.subheader("Filters")

    min_date, max_date = city_df['date'].min().date(), city_df['date'].max().date()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    pollutants_2 = ['PM2.5','PM10','NO','NO2','CO','Ozone','Nox','NH3']
    selected_polls = st.sidebar.multiselect("Pollutants", options=pollutants_2, default=['PM2.5','PM10','NO2'])
    
    plot_type= st.sidebar.selectbox("Plot type",[
        "Line Trends",
        "Bar Average",
        "Monthly Heatmap",
        "Correlation Heatmap"
    ],index=0)


    # --- Monthly aggregation ---
    city_df = city_df[(city_df['date'] >= start_date) & (city_df['date'] <= end_date)].copy()
    city_df['month_year'] = city_df['date'].dt.to_period('M')
    monthly_df = city_df.groupby('month_year')[selected_polls].mean().reset_index()
    monthly_df['month_year'] = monthly_df['month_year'].astype(str)

##############------MAP

    st.subheader("City Map with Wind direction")
    m1 = folium.Map(location=[20.2961, 85.8245], zoom_start=8)
    folium.Marker([20.2961,85.8245], popup="Bhubaneswar").add_to(m1)
    season_now = month_to_season(datetime.now().month)
    wind_dir = SEASON_WIND[season_now]
    folium.Marker([20.5,85.8], icon=folium.DivIcon(html=f"<div style='font-size:12pt;color:blue'>→ {wind_dir}</div>")).add_to(m1)
    st_folium(m1, width=700, height=300)

    pollutants_3=["None"]+pollutants_2
    polls_2 = st.sidebar.selectbox("Pollutant for recommendation", options=list(pollutants_3),index=0)

    ###plotssss

    if plot_type == "Line Trends":
        st.subheader(f"Monthly Line Trends - {selected_city}")
        fig, ax = plt.subplots(figsize=(10,5))
        for p in selected_polls:
            ax.plot(monthly_df['month_year'],monthly_df[p], marker='o', label=p)
        ax.set_xticks(range(0,len(monthly_df['month_year']),max(1,len(monthly_df)//12)))
        ax.set_xticklabels(monthly_df['month_year'][::max(1,len(monthly_df)//12)],rotation=45)
        ax.set_ylabel("Concentration");ax.legend()
        st.pyplot(fig)


    elif plot_type == "Bar Average":
        st.subheader(f"Average concentration by pollutant - {selected_city}")
        avg_val= monthly_df[selected_polls].mean().sort_values(ascending=False)
        fig,ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=avg_val.index, y=avg_val.values,ax=ax, palette="viridis")
        ax.set_ylabel("Mean Concentration"); ax.set_xlabel("Pollutant")
        st.pyplot(fig)

    elif plot_type == "Monthly Heatmap":
        st.subheader(f"Monthly Heatmap -{selected_polls}")
        pivot = monthly_df.set_index('month_year')[selected_polls].T
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(pivot, cmap="YlOrRd",annot=True,fmt=".1f",ax=ax)
        ax.set_xlabel("Month-Year");ax.set_ylabel("Pollutant")
        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        st.subheader(f"Correlation Between Pollutants — {selected_city}")
        corr = monthly_df[selected_polls].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax, center=0)
        st.pyplot(fig)
    

     ##########------RECOMMENDED---ACTIONS
    st.subheader("Recommended Actions (user-selected pollutant)")

    if polls_2 == "None":
        st.info("No pollutant selected. Please choose one in the sidebar to view recommended actions.")
    else:
        st.write(f"**Selected pollutant:** {polls_2}")
        rec_trees = mapping.get(polls_2, ["Local species"])
        st.write("Recommended trees:", ", ".join(rec_trees))

        if polls_2 in ["PM2.5", "PM10"]:
            st.write("""
            - **Action:**  
            1. Plant Banyan, Peepal, Neem, Mango, Jamun in urban roadsides, markets, schools, and residential belts.  
            2. Plant Tulsi hedges near households and temples.  
            3. Use layered planting: tall canopy trees + shrubs near roads to trap suspended particles.  
            4. Plant avenue rows along traffic-heavy corridors (2–3 rows of dense foliage trees).  

            - **Why it works:** Broad leaves, rough surfaces trap dust/soot; multiple canopy layers reduce PM concentration.
            """)

        elif polls_2 in ["NO2", "SO2","NOx","NO"]:
            st.write("""
            - **Action:**  
            1. Plant Neem, Pongamia (Karanja), Peepal, Banyan around industries, power plants, and highways.  
            2. Create buffer greenbelts (30–100 m wide) around emission zones.  
            3. Combine with evergreens like Ashoka to form year-round barriers.  

            - **Why it works:** High stomatal density trees absorb SO₂ & NO₂; dense canopies dilute and filter gases.
            """)

        elif polls_2 == "Ozone":
            st.write("""
            - **Action:**  
            1. Avoid Eucalyptus monocultures.  
            2. Plant Ficus, Jamun, Mango, Neem in city cores.  
            3. Use cooling green corridors to reduce precursor buildup.  

            - **Why it works:** Shade trees reduce local temperature; low-VOC trees minimize precursors.
            """)

        elif polls_2 == "CO":
            st.write("""
            - **Action:**  
            1. Plant dense roadside vegetation (Banyan, Peepal, Mango, Neem).  
            2. Create micro-forests in residential belts.  

            - **Why it works:** Trees with high leaf surface area help dilute localized vehicle exhaust.
            """)

        elif polls_2 == "NH3":
            st.write("""
            - **Action:**  
            1. Plant Neem, Moringa, Tulsi, Pongamia, and Vetiver near agricultural zones.  
            2. Use bio-shields around fertilizer-intensive areas.  

            - **Why it works:** These plants absorb ammonia and help manage nitrogen cycles.
            """)

        else:
            st.write("- Action: Plant dense trees and recommended greening; wear masks outdoors when needed.")

    st.write("---")
    st.caption("This app uses visual-based dominance. For production-grade systems, we'd add uncertainty estimates, more contextual features, and ML analysis.")
