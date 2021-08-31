## Datasets for the paper "Cross-Lingual Text Classificationnof Transliterated Hindi and Malayalam"

Please contact jkrishn2@gmu.edu to request the tweet datasets.

### Twitter Policy 
Tweets extraction procedure was per Twitter ToS and did not violate privacy policies of individual users. Also data shared includes only Tweet IDs in the public domain.

### New Dataset: Malayalam Movie Reviews (4 files)

Folder Location: ```data/movie_reviews/test```

Filenames: ```ml_pos```, ```ml_pos_ro```, ```ml_neg```, ```ml_neg_ro``` ('ro' representing romanized data)

Task - Sentiment Analysis: Select highly polar sentences from movie reviews from the news website samayam.com -https://malayalam.samayam.com/malayalam-cinema/movie-review/articlelist/48225004.cms

Labels: Positive = ```1```, Negative = ```0```

Guidelines/Rationale: Only select obviously polar sentences. Avoid neutral or ambiguous sentences.

- Annotator: One Native Speaker (College Graduate)
- Annotator Demographic: Kerala, India

Examples:
+: സമീപ കാലത്ത് കണ്ട ഫഹദിന്‍റെ ഏറ്റവും മികച്ച പ്രകടനങ്ങളിലൊന്ന്.
-: ഞാൻ സിനിമയിൽ വളരെ നിരാശനായിരുന്നു.

### New Dataset: Kerala Floods (1 file)

Folder Location: data/floods/test

Filename: kerala_floods.csv

Task - Relevancy Filtering: Select tweets that are related/relevant to Kerala Floods.

Tweets Scraped Timeline: 2018-08-08 to 2018-09-30.

Keywords: pralayam, vellapokkam, vellam, sahayam, durantham, veedukal, nasanashta, neenthal, bakshanam, vellathil, kollapettu, thozhilali, rakshapravarthanam, maranam.

Labels: Positive (Relevant) = ```1```, Negative (Irrelevant) = ```0```

Guidelines/Rationale: Label the tweets '1' that are relevant only to floods. Select only transliterated sentences.

Annotator(s): Two Native Speakers (College Graduates)
Annotator Demographic: Kerala, India

Examples:
+: Chavakkad Guruvayoor pradheshangalil bakshanam 
avishyamullavar thaazhe koduthitulla number il udane 
bendhapeduka.. Rajah  School chavakkad...!!!
-: Kadha Thudarunnu - Aaro Paadunnu Doorey

### New Dataset: North India Floods (1 file)

Folder Location: data/floods/test

Filename: north_india_floods.csv

Task - Relevancy Filtering: Select tweets that are related/ 
relevant to North India Floods.

Tweets Scraped Timeline: 2013-06-01 to 2013-07-30.

Keywords: madad, toofan, baarish, sahayta, floods, samay, suraksha, varsha, baad, gaya hai, hota hai.

Labels: Positive (Relevant) = ```1```, Negative (Irrelevant) = ```0```

Guidelines/Rationale: Label the tweets '1' that are relevant only to floods. Select only transliterated sentences.

Annotator(s): Two Fluent Speakers 
(College Graduates + Bilingual Proficiency)
Annotator Demographic: Kerala, India

Examples:
+: Reporter:- Aapka Kya Nuksaan Hua H? Flood survivor:- 
Mera Mard Beh Gaya. Ghar Toot Gaya. Khane Ko Kuch Nahi. 
Reporter:- Par Nuksaan Kya Hua?

-: johnson's baby lotion karay apke shishu ki komal 
towcha ki suraksha.  Haha twadi pehn da shishu. Haha.

### Existing Datasets (Will be provided in the github link - too large)

MOVIE REVIEWS (IMDB - Train/Validation):
- ```data/movie_reviews/imdb_train_val/train/en```
- ```data/movie_reviews/imdb_train_val/val/en```

MOVIE REVIEWS (Hindi - Test):
Folder Location: ```data/movie_reviews/test```
Filenames: ```hi_pos```, ```hi_pos_ro```, ```hi_neg```, ```hi_neg_ro```

APPEN (CRISIS - Train/Validation):
- ```data/floods/appen-train-val/train/en```
- ```data/floods/appen-train-val/val/en```

### Augmented Datasets
```ml``` := malayalam
```hi``` := hindi
```ro``` := romanized
```combo``` := en+tr+tl
