param nSong; 
param nPart;
param nGenre;
param M;

set song := {1..nSong};
set feature := {1..10};
set part := {1..nPart};
set genre := {1..nGenre};

param Actual {song, genre} >= 0;
param X {song, part, genre} >= 0;

var w {genre, part, feature} >= -1000, <= 1000;
var n {song, genre} >= -1000, <= 1000;
var b {genre} >= -1000, <= 1000; 

var z {feature} binary;

#var t{song} >= 0;
#var YPrime {song, genre} binary;
#var r {song} >= 0;
# +  (sum{f in feature} z[f]) + (0.5 * sum{g in genre, p in part, f in feature} w[g,p,f]^2)

minimize Objective: sum{g in genre, s in song} (-Actual[s,g] * n[s,g] + log(sum{f in feature} exp(n[s,f])) ) + (10 * sum{f in feature} z[f])  + (0.5 * sum{g in genre, p in part, f in feature} w[g,p,f]^2);

s.t. ct1 {g in genre, s in song}: n[s,g] == (sum{p in part, f in feature} (w[g,p,f] * X[s,p,f]) ) + b[g];

s.t. ct2 {g in genre, p in part, f in feature}: w[g,p,f] >= (-9999 * z[f]);

s.t. ct3 {g in genre, p in part, f in feature}: w[g,p,f] <= (9999 * z[f]);

#s.t. ct4 {g in genre, s in song}: t[s] >= n[s,g];

#s.t. ct5 {g in genre, s in song}: t[s] <= n[s,g] + (1-YPrime[s,g]);

#s.t. ct6 {s in song}: sum{g in genre} YPrime[s,g] == 1;

#s.t. ct7 {s in song}: r[s] == sum{g in genre} (YPrime[s,g] * Actual[s,g]);
