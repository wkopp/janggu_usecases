removeStart = function(df, pattern, type=NULL) {
  if(is.null(type)) {
    i = which(startsWith(as.character(df$labels), pattern))
  } else {
    i = which(startsWith(as.character(df$labels), pattern) & type == df$type)
  }
  
  df$labels[i] = substr(df$labels[i], nchar(pattern)+1, nchar(as.character(df$labels[i])))
  return(df)
}

removeEnd = function(df, pattern) {
  i = which(endsWith(as.character(df$labels), pattern))
  df$labels[i] = substr(df$labels[i], 1, nchar(as.character(df$labels[i])) - nchar(pattern))
  return(df)
}

# process labels, remove redundant parts
df = removeStart(df, "E003.")
df = removeStart(df, "E123.")
df = removeStart(df, "E124.")
df = removeStart(df, "E125.")
df = removeStart(df, "E126.")
df = removeStart(df, "E127.")
df = removeStart(df, "E128.")
df = removeStart(df, "E129.")
df = removeEnd(df, ".srt")
df = removeEnd(df, ".narrowPeak")
df = removeEnd(df, "UniPk")

df = removeStart(df, "wgEncodeAwgDnaseUw")
df = removeStart(df, "wgEncodeAwgDnaseDuke")
df = removeStart(df, "wgEncodeAwgTfbs")
df = removeStart(df, "duke")
df = removeStart(df, "Broad")
df = removeStart(df, "Haib")
df = removeStart(df, "Sydh")
df = removeStart(df, "Uchicago")
df = removeStart(df, "Uta")
df = removeStart(df, "Uw")

#itfs = which(df$type == "Tfbs")
#
#for (i in itfs) {
#  res = gregexpr("[A-Z]", df$labels[i])[[1]]
#  if (length(res) == 2) {
#    new_label =  substr(df$labels[i], res[2], nchar(df$labels[i]))
#  } else {
#    new_label =  substr(df$labels[i], res[2], res[3]-1)
#  }
#  df$labels[i] = new_label
#}


