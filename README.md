# sentiment-surge
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import warnings

# Configuration
warnings.filterwarnings('ignore')
load_dotenv()
pd.set_option('display.max_columns', None)

# Constants
TICKERS = ['TSLA', 'NVDA']
NEWS_SOURCES = {
    'Reuters': 'https://www.reuters.com/search/news?blob=',
    'Bloomberg': 'https://www.bloomberg.com/search?query='
}
LLM_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_THRESHOLD = 0.2  # Absolute score threshold for neutral classification
LOOKBACK_DAYS = 30  # How many days of historical data to analyze

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=LLM_MODEL,
    tokenizer=LLM_MODEL
)

class FinancialSentimentAnalyzer:
    def __init__(self):
        self.news_data = pd.DataFrame()
        self.price_data = pd.DataFrame()
        self.correlation_results = {}
    
    def scrape_news(self, ticker, days=7):
        """Scrape financial news for given ticker from multiple sources"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_articles = []
        
        for source, base_url in NEWS_SOURCES.items():
            try:
                search_url = f"{base_url}{ticker}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(search_url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Source-specific parsing (simplified - would need customization)
                if source == 'Reuters':
                    articles = soup.find_all('div', class_='search-result-indiv')
                    for article in articles:
                        title = article.find('h3').get_text(strip=True)
                        timestamp = article.find('span', class_='timestamp').get_text(strip=True)
                        content = article.find('p').get_text(strip=True)
                        all_articles.append({
                            'ticker': ticker,
                            'source': source,
                            'title': title,
                            'content': content,
                            'timestamp': timestamp,
                            'date': pd.to_datetime(timestamp).date()
                        })
                
                elif source == 'Bloomberg':
                    articles = soup.find_all('article', class_='search-result-story')
                    for article in articles:
                        title = article.find('a', class_='headline').get_text(strip=True)
                        timestamp = article.find('time')['datetime']
                        content = article.find('div', class_='summary').get_text(strip=True)
                        all_articles.append({
                            'ticker': ticker,
                            'source': source,
                            'title': title,
                            'content': content,
                            'timestamp': timestamp,
                            'date': pd.to_datetime(timestamp).date()
                        })
            
            except Exception as e:
                print(f"Error scraping {source} for {ticker}: {str(e)}")
        
        return pd.DataFrame(all_articles)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using LLM with threshold-based classification"""
        try:
            result = sentiment_analyzer(text[:512])[0]  # Truncate to model max length
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            
            if abs(score) < SENTIMENT_THRESHOLD:
                return 'neutral', score
            return ('positive' if score > 0 else 'negative'), score
        except:
            return 'neutral', 0
    
    def get_stock_prices(self, ticker, days=30):
        """Get historical stock prices for given ticker"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            data = data[['Close']].reset_index()
            data['ticker'] = ticker
            data['pct_change'] = data['Close'].pct_change() * 100
            data['date'] = pd.to_datetime(data['Date']).dt.date
            return data.drop(columns=['Date'])
        except Exception as e:
            print(f"Error fetching prices for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def process_ticker(self, ticker):
        """Full processing pipeline for a single ticker"""
        print(f"\nProcessing {ticker}...")
        
        # Step 1: Scrape recent news
        news_df = self.scrape_news(ticker, days=7)
        if news_df.empty:
            print(f"No news found for {ticker}")
            return
        
        # Step 2: Perform sentiment analysis
        news_df[['sentiment', 'sentiment_score']] = news_df['content'].apply(
            lambda x: pd.Series(self.analyze_sentiment(x))
        )
        
        # Step 3: Get price data
        price_df = self.get_stock_prices(ticker, days=LOOKBACK_DAYS)
        if price_df.empty:
            print(f"No price data found for {ticker}")
            return
        
        # Step 4: Merge and aggregate data
        merged_df = pd.merge(
            news_df,
            price_df,
            on=['date', 'ticker'],
            how='left'
        ).dropna(subset=['pct_change'])
        
        if merged_df.empty:
            print(f"No overlapping news and price data for {ticker}")
            return
        
        # Step 5: Calculate daily sentiment metrics
        daily_sentiment = merged_df.groupby('date').agg({
            'sentiment_score': ['mean', 'count'],
            'pct_change': 'first'
        }).reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment', 'news_count', 'price_change']
        
        # Step 6: Calculate correlations
        corr, p_value = pearsonr(daily_sentiment['avg_sentiment'], daily_sentiment['price_change'])
        mape = mean_absolute_percentage_error(
            daily_sentiment['price_change'],
            daily_sentiment['avg_sentiment'] * 100  # Scale sentiment to match percent change
        )
        
        self.correlation_results[ticker] = {
            'pearson_corr': corr,
            'p_value': p_value,
            'mape': mape,
            'news_count': len(news_df),
            'days_analyzed': len(daily_sentiment)
        }
        
        # Step 7: Visualization
        self.plot_sentiment_analysis(ticker, daily_sentiment)
        
        return daily_sentiment
    
    def plot_sentiment_analysis(self, ticker, data):
        """Visualize sentiment vs price movement"""
        plt.figure(figsize=(12, 6))
        
        # Plot price change
        ax1 = plt.gca()
        ax1.plot(data['date'], data['price_change'], 'b-', label='Price Change (%)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price Change (%)', color='b')
        ax1.tick_params('y', colors='b')
        
        # Plot sentiment on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(data['date'], data['avg_sentiment'], color='r', alpha=0.3, label='Avg Sentiment')
        ax2.set_ylabel('Average Sentiment Score', color='r')
        ax2.tick_params('y', colors='r')
        
        plt.title(f'{ticker} - Price Change vs News Sentiment')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Add correlation info to plot
        corr_info = self.correlation_results[ticker]
        plt.annotate(
            f"Pearson r: {corr_info['pearson_corr']:.2f} (p={corr_info['p_value']:.3f})\nMAPE: {corr_info['mape']:.2f}",
            xy=(0.02, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.show()
        
        # Plot sentiment distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(data['avg_sentiment'], bins=20, kde=True)
        plt.title(f'{ticker} - Distribution of Daily Average Sentiment Scores')
        plt.xlabel('Average Sentiment Score')
        plt.ylabel('Frequency')
        plt.show()
    
    def generate_insights(self):
        """Generate actionable insights from analysis"""
        print("\n=== Investment Insights ===")
        
        for ticker, results in self.correlation_results.items():
            print(f"\n{ticker} Analysis:")
            print(f"• News articles analyzed: {results['news_count']}")
            print(f"• Pearson correlation (sentiment vs price): {results['pearson_corr']:.2f}")
            print(f"• Correlation p-value: {results['p_value']:.3f}")
            print(f"• Mean Absolute Percentage Error: {results['mape']:.2f}")
            
            # Generate simple trading signal based on correlation
            if results['p_value'] < 0.05:
                if results['pearson_corr'] > 0.3:
                    print("✅ STRONG POSITIVE correlation detected - News sentiment appears to lead price movements")
                    print("   Consider: Buying when positive sentiment increases, selling when negative")
                elif results['pearson_corr'] < -0.3:
                    print("⚠️ STRONG NEGATIVE correlation detected - Contrarian pattern observed")
                    print("   Consider: Buying on negative news, selling on positive news")
                else:
                    print("🔍 Moderate correlation detected - Sentiment may be one factor among many")
            else:
                print("📊 No statistically significant correlation found - Sentiment may not predict price movements")
            
            # MAPE interpretation
            if results['mape'] < 50:
                print(f"✓ Model error (MAPE) is relatively low ({results['mape']:.1f}%) - Reasonable predictive power")
            else:
                print(f"⚠️ High model error ({results['mape']:.1f}%) - Use sentiment with caution")

# Main execution
if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    
    # Process each ticker
    for ticker in TICKERS:
        analyzer.process_ticker(ticker)
    
    # Generate insights
    analyzer.generate_insights()
