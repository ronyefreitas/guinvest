# Guinvest
This project is a part of my personal Data Science portfolio and uses Streamlit as a graphic interface.

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is to create a simple end-user tool which allows graphic analysis of the Brazilian stocks market, as well as predicting future closing prices through ARIMA based models.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Time series analysis

### Technologies
* Python
* Streamlit
* pmdarima
* yfinance
* Pandas
* Plotly
* BeautifulSoup

## Project Description
<!-- (Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here) -->
Guinvest is an application uses [Fundamentus](https://www.fundamentus.com.br/resultado.php) website as its source for Brazilian stocks market tickers, which are gathered through web-scraping using BeautifulSoup, as well as yfinance library, that provides a real time connection with [Yahoo! Finance](https://finance.yahoo.com/) from where the stocks closing prices information are pulled. Once the data is loaded, the application enables the user to visualize through Plotly graphs, perform closing prices predictions based on ARIMA model and evaluate the model performance.



## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Make sure you have Python 3.7> installed in your computer.
3. Run the install-libraries.py script to install all the required libraries.
4. To start the application, inside Guinvest folder, run the following command on termimal: $ streamlit run guinvest.py
5. A web browser will open and you will see the application
![Start screen](/screenshots/start-screen.png)
6. Select all the tickers of the stocks you want to analyze on the box 'Tickers' under 'My Stocks' section. Then, select the start date inputter 'From' and the end date at the date inputter 'Until'. The default period is set to be the of one year and you should mind that for fair predictions the ARIMA model requires at least 90 observations (3 months, since we are dealing with daily closing prices).
![Choosing tickers](/screenshots/choosing-tickers.png)
7. Select the ticker you want to run a deeper model evaluation, as well as the evaluation period in days under the 'Evaluating models' sections. Mind that the longer the period, the more processing intensive will be the app.
![Model evaluation](/screenshots/model-evaluation.png)

## Contact
ronye.freitas@gmail.com
