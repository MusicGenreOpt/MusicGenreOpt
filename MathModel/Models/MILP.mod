param nSong; 
param nPart;
param nGenre;
param M;

set song := {1..nSong};
set genre := {1..nGenre};
set part := {1..nPart};

var P{part} >= 0;
var Y {song, genre} >= 0;
var YPrime {song, genre} >= 0 binary;
var r {song} >= 0;
#var u {song} binary;
var t{song} >= 0;
#var p_max >= 0;
#var p_min >= 0;


#param set_part {p in part} = if u[p] = 1 then p;
#param P {part};
param Actual {song, genre};
	
param X {song, part, genre};

#minimize Objective: nSong - (sum{s in song} (r[s])) + (0.9 * sum{s in song} (u[s]));
minimize Objective: nSong - (sum{s in song} (r[s]));

s.t. ct1 {g in genre, s in song}: sum{p in part} (P[p] * X[s,p,g]) == Y[s,g];

#s.t. ct2 {s in song}: sum{p in part} (P[p]) == 1;

s.t. ct3 {g in genre, s in song}: t[s] >= Y[s,g];

s.t. ct4 {g in genre, s in song}: t[s] <= Y[s,g] + (1-YPrime[s,g]);

s.t. ct5 {s in song}: sum{g in genre} YPrime[s,g] == 1;

s.t. ct6 {s in song}: r[s] == sum{g in genre} (YPrime[s,g] * Actual[s,g]);

#s.t. ct7 {p in part}: p_max >= P[p];

#s.t. ct8 {p in part}: -p_min >= -P[p];

#s.t. ct9: p_max - p_min <= 0.05;

#s.t. ct10 {p in part}: P[p] <= u[p];

#s.t. ct11: -p_min <= -0.000001;
#s.t. ct7 {s in song}: M * t[s] + u[s] >= 1;
