topic_code={ 
            'Announced Layoffs': "(Topic:REORG OR Topic:LAYOFS)",
            'Anti-Competition Controversy': "(Topic:MONOP OR Topic:ACB)",
            'FDA Warning Letter': "(FDA AND ( warning OR Inspect OR Letter ))",
            'Wages or Working Condition Controversies':"(Topic:DISP)",
            'Product Recall':"(Topic:RECLL)",
            'Management Departures': "(Retire OR Topic:JOB)",
            'Shareholder Rights Controversies':"(Topic:CLASS)",
            "Diversity and Opportunity Controversies":"(Topic:RACR AND Topic:JOB)",
            # "Freedom of Association Controversies":"(Topic:HRGT AND Topic:JOB)",
            "Human Rights Controversies":"(Topic:CIV OR Topic:HRGT)"
            }

topic_code={ 
'Announced Layoffs': "M:222", #(Topic:REORG OR Topic:LAYOFS)", #M:222
'Anti-Competition Controversy': "M:21W", #(Topic:MONOP OR Topic:ACB)", # M:21W
'Wages or Working Condition Controversies':"M:225",#(Topic:DISP)", # M:225
'Consumer Complaints':"(M:21I)",
'Drug Disapproval':"(M:21Q)",
'Strikes':"M:224",
'Management Departures': " M:223", #(Retire OR Topic:JOB)", # M:223
'Diversity and Opportunity Controversies':"M:221", #(Topic:RACR AND Topic:JOB)", # M:221
'Human Rights Controversies':"M:220", #(Topic:CIV OR Topic:HRGT)", # M:220
'Intellectual Property Controversies':"(M:21S)",
'Shareholder Rights Controversies':"(M:21F)",
# 'Freedom of Association Controversies':"M:21Z", #(Topic:HRGT AND Topic:JOB)", # M:21Z
}

topic_list = list(topic_code.keys())

n_topics = len(topic_list)

id2label = {idx:label for idx, label in enumerate(topic_list)}
label2id = {label:idx for idx, label in enumerate(topic_list)}

