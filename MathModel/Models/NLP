param nSong; 
param nPart;
param nGenre;

set song := {1..nSong};
set genre := {1..nGenre};
set part := {1..nPart};

var Y {song, genre} >= 0;
var YPrime {song, genre} >= 0 binary;
var r {song} >= 0;
var t{song} >= 0;

param P {part};
param Actual {song, genre};
param X {song, part, genre};

minimize Objective: sum{g in genre, s in song} ( -Actual[s,g] * log(exp(Y[s,g]))

minimize Objective: nSong - (sum{s in song} (r[s]));

s.t. ct1 {g in genre, s in song}: sum{p in part} (P[p] * X[s,p,g]) == Y[s,g];

s.t. ct2 {s in song}: sum{p in part} (P[p]) == 1;

s.t. ct3 {g in genre, s in song}: t[s] >= Y[s,g];

s.t. ct4 {g in genre, s in song}: t[s] <= Y[s,g] + (1-YPrime[s,g]);

#s.t. ct5 {s in song}: sum{g in genre} YPrime[s,g] == 1;

#s.t. ct6 {s in song}: r[s] == sum{g in genre} (YPrime[s,g] * Actual[s,g]);
