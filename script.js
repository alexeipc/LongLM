let arr = [[ "C", "B", "D", "BD", "A", "C", "C", "B", "C", "CD", "B", "ABC", "C", "B" ],
[ "B", "AC", "C", "A", "B", "A", "B", "A", "AC", "A", "ABC", "ABCD" ],
[ "D", "ACD", "C", "ABC", "A", "ACD", "B", "A", "AC" ],
[ "ABCD", "AB", "ABC", "AC", "D", "ACD", "AB", "BCD", "ACD", "AB" ],
[ "BC", "ABC", "D", "C", "ACD", "ABD", "AD", "AB", "B", "A", "A", "ABC", "D" ],
[ "A", "ABC", "A", "D", "B", "CD", "AB", "ABD", "ABC", "D", "ABC", "AB", "A" ],
[ "ABC", "ABC", "ABC", "ABD", "ABC", "ABC", "ABC", "ABCD", "ABC", "AB", "BD", "AB" ],
[ "B", "C", "C", "B", "C", "B", "A", "A", "AD", "ABC" ],
[ "ABD", "ACD", "A", "C", "ACD", "A", "ABD", "D", "A", "B" ],
[ "C", "B", "C", "C", "D", "D", "ABD", "B", "C", "C", "BCD", "BCD" ],
[ "BCD", "ABCD", "ACD", "BC", "BCD", "C", "C", "BD", "BCD", "ABC", "C", "A", "A", "B" ],
[ "AD", "B", "A", "A", "B", "ABD", "BCD" ],
[ "B", "A", "AC", "A", "AC", "B", "A", "D", "BCD", "ACD", "BC", "CD", "AC" ],
[ "ABC", "ACD", "ABC", "BC", "ABC", "ABD", "C", "ABD", "ABC", "D" ],
[ "C", "B", "A", "D", "A", "B", "BD", "A", "B", "A", "BD", "AC", "BD" ]]


quesions = 0
a = 0
score = 0

for(arr1 of arr) {
    for (item of arr1) {
        quesions += 1
        // f1 score with arr containing ground truth and 'a' is answer
        let answers = item.split("")
        tp = 0
        fn = 0
        for(answer of answers) {
            if (answer != "A") {
                fn += 1
            }
            else {
                tp += 1
            }
        }

        f1 = 2*tp / (2*tp + fn)
        score += f1
    }
}

console.log("Total Questions: " + quesions)
console.log("Total A Answers: " + a)
console.log("Total Score: " + score)
console.log("Average Score: " + score / quesions)