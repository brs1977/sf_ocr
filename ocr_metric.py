import fastwer

def score(df,char_level):
  sf_no_mean = df.apply(lambda row: fastwer.score_sent(row.sf_no_hypo, row.sf_no_ref, char_level),axis=1).mean()
  sf_date_mean = df.apply(lambda row: fastwer.score_sent(row.sf_date_hypo, row.sf_date_ref, char_level),axis=1).mean()

  buyer_inn_mean = df.apply(lambda row: fastwer.score_sent(row.buyer_inn_hypo, row.buyer_inn_ref, char_level),axis=1).mean()
  buyer_kpp_mean = df.apply(lambda row: fastwer.score_sent(row.buyer_kpp_hypo, row.buyer_kpp_ref, char_level),axis=1).mean()
  seller_inn_mean = df.apply(lambda row: fastwer.score_sent(row.seller_inn_hypo, row.seller_inn_ref, char_level),axis=1).mean()
  seller_kpp_mean = df.apply(lambda row: fastwer.score_sent(row.seller_kpp_hypo, row.seller_kpp_ref, char_level),axis=1).mean()

  return sf_no_mean,sf_date_mean,buyer_inn_mean,buyer_kpp_mean,seller_inn_mean,seller_kpp_mean
