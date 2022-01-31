curl 'https://ucsdcloud-my.sharepoint.com/personal/kawang_ucsd_edu/_layouts/15/download.aspx?UniqueId=cdecee2e%2Dd1a3%2D4f0d%2D86d7%2Deb26777b00b3' \
-X 'GET' \
-H 'Referer: https://ucsdcloud-my.sharepoint.com/personal/kawang_ucsd_edu/_layouts/15/onedrive.aspx' \
-H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
-H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15' \
--compressed --output file.tar.gz
# curl 'https://ucsdcloud-my.sharepoint.com/personal/s1bhavsa_ucsd_edu/_layouts/15/download.aspx?UniqueId=fa92f1d6%2Ddd69%2D4037%2D9a1a%2De50286326d06' \
#  -H 'authority: ucsdcloud-my.sharepoint.com' \
#  -H 'sec-ch-ua: "Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"' \
#  -H 'sec-ch-ua-mobile: ?0' \
#  -H 'upgrade-insecure-requests: 1' \
#  -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36' \
#  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9' \
#  -H 'service-worker-navigation-preload: true' \
#  -H 'sec-fetch-site: same-origin' \
#  -H 'sec-fetch-mode: navigate' \
#  -H 'sec-fetch-dest: iframe' \
#  -H 'referer: https://ucsdcloud-my.sharepoint.com/personal/s1bhavsa_ucsd_edu/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly91Y3NkY2xvdWQtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwvczFiaGF2c2FfdWNzZF9lZHUvRWpGVU5lazA0UDVHang5UWFzc0dGOThCWm5lWTVUTk8tZDMzWDZ0S1h1RUZiUT9ydGltZT00M3NTWmJqZDJFZw&id=%2Fpersonal%2Fs1bhavsa%5Fucsd%5Fedu%2FDocuments%2Fhypersim' \
#  -H 'accept-language: en-US,en-IN;q=0.9,en-GB;q=0.8,en;q=0.7' \
#  -H 'cookie: MicrosoftApplicationsTelemetryDeviceId=b6dfcb1a-5195-643a-a683-914d06dcf293; MicrosoftApplicationsTelemetryFirstLaunchTime=1614716344702; rtFa=r672Hm2eyhaR/NwiNDCrDO5SIxFJDP4myAYtMbm+ss0mQUU3RkMwNEMtOUUxMi00MTU0LThBN0EtQzQyRkY0MTVFQTI3IzEzMjU5MDQ0MjcxNDU2NjkyMSM2MkUxQUY5Ri0yMDUyLTAwMDAtN0YyQS0zNTQ0MDhEMURBMEZsC1Bhrq+vFrRcnq6+pBlg5D9aJfoiY9xX749RJSyOVu8k8kfyK7oYHf+HwYeOyedfyKu2umJJS459xOSSxxkkoX60gmmaVWpXsN2/Z4+pWrSHVBdty+DMbcHl7ftPJjpsdQi+SmKs35uDxuUfcys2HrhbIFIXVtOns/OvaUSubKn1uiQIM+RTDgCnATmTugURiOA32e1kGDrApag3097bVWJlT5QOAbVGgbK1O9gLfycyo5c11qcuvk65cry1wuVKURNu83CCiPLgv6KdkixKNYaDW6IdSUzoFNE/JYIbgW4rzjFvuj5WP+vIHYUnv9qjmkDRv0eDoNlRxpbDOKxWfQAAAA==; CCSInfo=My80LzIwMjEgNTo1NjoxMCBBTV24udbEtQS7Olxqp4QiatThG2iM+K/I+sqtd19utsEjd0mie3Q2ErJaFDJM5ds9W4YOS8t7x2Kn04afSai0rW5cxLIDYt+FcVG+b5q3NV46CBpyjNglHOyfSvZVGUObyZ4RfKUe28oZbUKrGSy7dR+z3tEQleoiVWPqXcl4jFf+wPMrHoKy+1Ro+acTOKL21OPUZak0BMWubPHSYzTMDJHi5FCPLeVedAhn8SLVw+YjrVMAEKnJHLiuyvc07OgNDzG+cUjbBq10Ka2fhT2KUqgpt3ACqkcO9tEKvcz9g4T9135e3zy7LNt20VdKW8q9L7nF5szb1Kf9QyUrx8dIOTkTAAAA; WSS_FullScreenMode=false; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjgsMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwYjEwNDZiNzlAbGl2ZS5jb20sMCMuZnxtZW1iZXJzaGlwfGFkaGFyYXNrQHVjc2QuZWR1LDEzMjU5MDQ0MjU1MDAwMDAwMCwxMzIzMTUyODUxNTAwMDAwMDAsMTMyNTk2MjE5MzE2Nzg2NTUzLDY2Ljc1LjI1My4xODEsMyxhZTdmYzA0Yy05ZTEyLTQxNTQtOGE3YS1jNDJmZjQxNWVhMjcsLGU1MDVkNjc3LTMzYjMtNDU2YS1hYWQ3LTM4OGUyZjhiMzRiYixmZDdiMTZkMy0zMGJhLTQyZGMtYjQ2ZS02NmRmM2M0YzI3NWYsMWE1ZmNiMTktNmUzNy00NzNmLWJhYzYtZjliNDIxYzczNDZmLCwwLDEzMjU5MTkzNTMxNjYzMDg3MSwxMzI1OTQ0OTEzMTY2MzA4NzEsLCxleUo0YlhOZlkyTWlPaUpiWENKRFVERmNJbDBpZlE9PSwyNjUwNDY3NzQzOTk5OTk5OTk5LDEzMjU5MDQ0MjcxMDAwMDAwMCw4OTdjZTdkZi01NWQ5LTQzNjgtOWZiYy04Zjc0YmQ3ZWRkMmQsYWJXRVdXcHo4elFkR1lhZUt0L2ZxWHk5ZmVueXdva2JHQW5yeXlzYm93ZFVZZGtDMk0wRTYxQXQrUjBqVG9Ma1BMdkxmNG5ybHRsV2hvUW1kbkxEZ05WejV5NUlDTTRqM2NiZWVZZllKRU93V2tzWGhrc0pTVEozVHFnNDdaZlZWTU9teUh0b1g2T0V1UDkxdm0zTFFGdTRxakp2ZjdyRWQyQXo3VW5mWVNRcGtUMVhTN1pPc1RuQVd2azU0MU5IanNscU1xY01qL2Fkai9STkhJMlRMSjRwNTRVajFDUzdGa3V6MGZXMzlGSjdaTTZBMW83YXVPeWxZNDk3bXJ4SWZSNjJxWXMxT3U4VHJiVVBlS2daSS8ySTMzQkRpbEVBdzNreWcxWWRremxFTUtaeTNNQUM5UUhVN3htUDZaN2trS3RiRGIwSFBjd2tGZmt5UWRGTzVBPT08L1NQPg==' \
tar -xzvf file.tar.gz
rm file.tar.gz
