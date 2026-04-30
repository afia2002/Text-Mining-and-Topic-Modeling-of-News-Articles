# Load required libraries
library(readr)
library(dplyr)
library(tm)
library(SnowballC)
library(tokenizers)
library(wordcloud)
library(RColorBrewer)
library(topicmodels)
library(ggplot2)
library(tidyr)
library(tidytext)

# 1. Load CSV file

articles <- read_csv("dailystar_articles.csv")
cat("CSV data loaded successfully. Number of articles:", nrow(articles), "\n")

# 2. Text Preprocessing
corpus <- VCorpus(VectorSource(articles$article_text))
custom_stopwords <- c("said", "also", "will", "today")

check_empty <- function(corpus, step) {
  lengths <- sapply(corpus, function(x) nchar(trimws(x$content)))
  if (any(lengths == 0)) {
    cat("Warning: Empty documents detected after", step, "\n")
    cat("Document indices:", which(lengths == 0), "\n")
  } else {
    cat("No empty documents after", step, "\n")
  }
}

corpus <- tm_map(corpus, content_transformer(tolower))
check_empty(corpus, "lowercase")
corpus <- tm_map(corpus, content_transformer(function(x) gsub("[^a-z ]", " ", x)))
check_empty(corpus, "clean symbols")
corpus <- tm_map(corpus, removeWords, stopwords("english"))
check_empty(corpus, "remove English stopwords")
corpus <- tm_map(corpus, removeWords, custom_stopwords)
check_empty(corpus, "remove custom stopwords")
corpus <- tm_map(corpus, stripWhitespace)
check_empty(corpus, "whitespace")
corpus <- tm_map(corpus, content_transformer(function(x) wordStem(x, language = "en")))
check_empty(corpus, "stemming")

# Remove empty docs
corpus <- corpus[sapply(corpus, function(x) nchar(trimws(x$content)) > 0)]
cat("Final corpus size:", length(corpus), "documents\n")

cat("\nPreprocessing Example:\n")
cat("Original:", articles$article_text[1], "\n")
cat("Cleaned:", corpus[[1]]$content, "\n")

# 3. Exploratory Text Analysis
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
word_freqs <- sort(rowSums(m), decreasing = TRUE)
df <- data.frame(word = names(word_freqs), freq = word_freqs)

# Word Cloud
dev.new()
wordcloud(words = df$word, freq = df$freq, min.freq = 2, max.words = 100,
          random.order = FALSE, colors = brewer.pal(8, "Dark2"))
cat("Word cloud displayed\n")

# Bar chart
top_20 <- head(df, 20)
dev.new()
ggplot(top_20, aes(x = reorder(word, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words", x = "Words", y = "Frequency") +
  theme_minimal()
cat("Top 20 word bar chart displayed\n")

cat("\nTop frequent words highlight political, economic, and social themes.\n")

# 4. Topic Modeling
dtm <- DocumentTermMatrix(corpus)
if (nrow(dtm) == 0 || ncol(dtm) == 0) stop("DTM is empty.")

# LDA model
lda_model <- LDA(dtm, k = 4, control = list(seed = 42))

topics <- tidy(lda_model, matrix = "beta")

top_terms_df <- topics %>%
  group_by(topic) %>%
  slice_max(order_by = beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Save to CSV
write.csv(top_terms_df, "top_terms_per_topic.csv", row.names = FALSE)
cat("Top terms per topic saved to 'top_terms_per_topic.csv'\n")

# Display Top Words per Topic (Bar Plot)
dev.new()
ggplot(top_terms_df, aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_bar(stat = "identity") +
  facet_wrap(~topic, scales = "free", ncol = 2) +
  coord_flip() +
  labs(title = "Top 10 Words per Topic", x = "Words", y = "Beta") +
  theme_minimal()
cat("Top words per topic displayed in R plot window\n")

# Document-topic distribution
doc_topics <- tidy(lda_model, matrix = "gamma")


