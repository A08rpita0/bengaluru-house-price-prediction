# Data

The model is trained on the Kaggle dataset **"Bengaluru House price data"**
(13,320 property listings; columns `area_type, availability, location, size, society,
total_sqft, bath, balcony, price`, with `price` in lakhs of INR).

The CSV is not committed (see `.gitignore`). Training expects it at
`data/Bengaluru_House_Data.csv`.

## How to fetch

Either download it from Kaggle ("Bengaluru House price data") and save it under the
name above, or use the public mirror that was used for the committed model:

```bash
curl -L -o data/Bengaluru_House_Data.csv \
  https://raw.githubusercontent.com/codebasics/py/master/DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv
```

`models/metadata.json` records the SHA-256 of the file used for training
(`data.sha256`), so you can confirm you have the same data:

```bash
python -c "import hashlib;print(hashlib.sha256(open('data/Bengaluru_House_Data.csv','rb').read()).hexdigest())"
```
