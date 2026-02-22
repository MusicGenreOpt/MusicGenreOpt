###############################################################################
# MLR-V2.mod
###############################################################################

param nSong;
param nPart;
param nGenre;
param M;

set song    := 1..nSong;
set part    := 1..nPart;
set genre   := 1..nGenre;

# Fixed feature set as in your model
set feature := 1..10;

param Actual {song, genre} >= 0;
param X {song, part, genre} >= 0;

# Big-M parameter (used in feature selection constraints)
param M >= 0 default 9999;

var w {genre, part, feature} >= -1000, <= 1000;
var n {song, genre} >= -1000, <= 1000;
var b {genre} >= -1000, <= 1000; 

var z {feature} binary;

minimize Objective:
    - sum {s in song, g in genre}
        Actual[s,g] * ( n[s,g] - log( sum {c in genre} exp(n[s,c]) ) )
    + 0.5 * sum {g in genre, p in part, f in feature} (w[g,p,f]^2)
    + 100 * sum {f in feature} z[f];

s.t. ct1 {s in song, g in genre}:
    n[s,g] = sum {p in part, f in feature} w[g,p,f] * X[s,p,f] + b[g];

s.t. ct2 {g in genre, p in part, f in feature}:
    w[g,p,f] >= -M * z[f];

s.t. ct3 {g in genre, p in part, f in feature}:
    w[g,p,f] <=  M * z[f];

# Eq.(5) in paper: equal weights across parts
s.t. ctV2 {g in genre, f in feature, i in part, j in part: i < j}:
    w[g,i,f] = w[g,j,f];
