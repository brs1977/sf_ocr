curl -F "file=@input/1.PDF" "127.0.0.1:9095/ocr"
curl "127.0.0.1:9095/progress/1903f632-9855-47e0-9bfd-23cebc455724"
curl "127.0.0.1:9095/result/b40d7134-5bb8-41d9-97d4-810ba446fc21" --output 1.zip
