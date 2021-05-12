"""Hackerrank Practice.
May 2, 2021

1. Linux Shell - 64 exercises
- mostly easy and medium
=============================================================================

11 Bash                       10 / 11
32 Text Processing            32 / 32
 8 Arrays in Bash              5 /  8
14 Grep Sed Awk                6 / 14
---                           ------
65                            53 / 65 77%
"""

# %% Linux Shell - BASH Language - Let's Echo
# Write a bash script that prints the string "HELLO".
# quick tutorial http://www.panix.com/~elflord/unix/bash-tute.html
# echo "Greetings $USER, your current working directory is $PWD"

# echo 'HELLO'

# %% Linux Shell - BASH Language - A Personalized Echo
# https://www.howtogeek.com/261591/how-to-create-and-run-bash-shell-scripts-on-windows-10/
# read name
# echo "Welcome $name"
# Looping and skipping
for number in {1..99..2}
do
    echo $number
done
e
for number in {1..50}
do
    echo $number
done


read x
read y

echo $((x + y))
echo $((x - y))
echo $((x * y))
echo $((x / y))


read X
read Y
if [ "$X" == "$Y" ]; then
  echo "X is equal to Y"
if [ "$X" -gt "$Y" ]; then
  echo "X is greater than Y"
else
  echo "X is less than Y"
fi

# Comparing numbers

# First try : elegant, but for some reason yields errors
# probably because all three comparisons are evaluated
read x
read y
[[ $x -gt $y ]] && echo 'X is greater than Y'
[[ $x -eq $y ]] && echo 'X is equal to Y'
[[ $x -lt $y ]] && echo 'X is less than Y'

# works
read x
read y
if (($x > $y)); then
    echo "X is greater than Y"
elif (($x == $y)); then
    echo "X is equal to Y"
else
    echo "X is less than Y"
fi


# Getting started with conditionals

# this doesn't work for some reason
read y
if (($y == N)); then
    echo "NO"
elif (($y == n)); then
    echo "NO"
else
    echo "YES"
fi

# this works
read char
echo -e "YES\nNO\n" | grep -i $char


# More on conditionals
# WORKS!
read x
read y
read z
if (($x == $y && $x == $z)); then
    echo 'EQUILATERAL'
elif (($x == $y && $x != $z)); then
    echo 'ISOSCELES'
elif (($z == $y && $x != $z)); then
    echo 'ISOSCELES'
else
    echo 'SCALENE'
fi

# WORKS!
read x
read y
read z
if (($x == $y && $x == $z)); then
    echo 'EQUILATERAL'
elif (($x == $y && $x != $z )) || (($z == $y && $x != $z)); then
    echo 'ISOSCELES'
else
    echo 'SCALENE'
fi

# Linux Shell - BASH - Arithmetic operations
# read in math expression and round result to 3 decimal places
# bc = basic calculator
# https://linuxhint.com/what-is-bc-bash-script/
printf "%.3f" "$(bc -l)"

# Linux Shell - BASH - Compute the Average
# read in array, compute average
read n
arr = ($(cat))
arr=${arr[*]}
printf "%.3f" $(echo $((${arr// /+}))/$n | bc -l)


cut -c3  # prints out 3rd character of each input line of input text file
cut -c2,7  # prints out 2nd and 7th characters
cut -c2-7  # prints out 2nd thru 7th characters
cut -c1-4  # prints chars 1-4
cut -c13-  # prints char 13 onward

cut -f1-3  # prints fields 1-3 of tab separated text

cut -w4  # prints char 13 onward

cut -d " " -f4  # print 4th word in line, delimiter is space
cut -d " " -f 1-3  # print out words 1-3
cut -f 2- # print out fields 2 to end (tab is default delimiter)

head -n 20  # prints first 20 lines of input
head -c 20  # prints first 20 characters of input

# print lines 12-22 inclusive; + starts counting from beginning
head -n 22 | tail -n +12
tail -n 20  # porint last 20 lines
tail -c 20  # porint last 20 chars

tr "()" "[]"  # replace () with []s
tr "\n" "\t"  # replace newline with tab

tr -d [:lower:]
tr -d a-z
tr -s " "

sort
sort -r
sort -n  # numbers
sort -nr   # reverse, numbers
sort -t$'\t' -k2 -rn   # sort by second column, separator is tab, descending
sort -t$'\t' -k2 -n   # sort by second column, separator is tab, ascending
sort -t$'|' -k2 -rn   # sort by second column, separator is tab, descending

uniq  # gets rid of consecutive duplicates

uniq -c|cut -b 7-
uniq -ic|cut -b 7-  # csae insensitive
uniq -u  # only print those that are not followed by exact duplicate
# =============================================================================
# Usage: cut [OPTIONS] [FILE]...
# Print selected fields from each input FILE to stdout
# 	-b LIST	Output only bytes from LIST
# 	-c LIST	Output only characters from LIST
# 	-d CHAR	Use CHAR instead of tab as the field delimiter
# 	-s	Output only the lines containing delimiter
# 	-f N	Print only these fields
# 	-n	Ignored
# =============================================================================
paste -s -d";"
# where -s = serial -d";" = delimiter separated by ;
paste -sd ';;\n'
paste -sd $'\t\t\n'
paste - - -

# %% Linux Shell - Text Processing -

# %% Linux Shell - Arrays in Bash -

# read in elements of array and then print out with space separating
a=($(cat))
echo ${a[@]}
# or
xargs
# read in elements of array and then print out with space separating

# slice an array, elements 3 through 7
arr=($(cat))
echo ${arr[@]:3:5}

# display array three times with space separator
array=($(cat))
Total=("${array[@]}" "${array[@]}" "${array[@]}")
echo ${Total[@]}

# display element 3
array=($(cat))
echo ${array[@]:3:1}
# or
a=($(cat))
echo ${a[3]}

# count elements
arr=($(cat))
echo ${#arr[@]}
# or
wc -l

# =============================================================================
# 'Sed' command #3
# Sed - An Introduction and a tutorial
https://www.grymoire.com/Unix/Sed.html#uh-10a

# The TLDP Guide
https://tldp.org/LDP/abs/html/x23170.html

# Some Practical Examples
https://www.folkstalk.com/2012/01/sed-command-in-unix-examples.html

# A StackOverflow question on a slightly modified version of this task where
# the solution involves backreferences.
https://stackoverflow.com/questions/2232200/regular-expression-in-sed-for-masking-credit-card

# A ttuorial from TheGeekStuff detailing the use of groups and backreferences.
https://www.thegeekstuff.com/2009/10/unix-sed-tutorial-advanced-sed-substitution-examples/

# Backreferences
https://www.thegeekstuff.com/2009/10/unix-sed-tutorial-advanced-sed-substitution-examples/

# Substitute the first occurrence of 'editor' with 'tool'.
echo "My favorite programming editor is Emacs. Another editor I like is Vim." | sed -e s/editor/tool/
# My favorite programming tool is Emacs. Another editor I like is Vim.

# Substitute all the occurrences of 'editor' with 'tool'.
echo "My favorite programming editor is Emacs. Another editor I like is Vim." | sed -e s/editor/tool/g
# My favorite programming tool is Emacs. Another tool I like is Vim.

# Substitute the second occurrence of 'editor' with 'tool'.
echo "My favorite programming editor is Emacs. Another editor I like is Vim." | sed -e s/editor/tool/2
# My favorite programming editor is Emacs. Another tool I like is Vim.

# Highlight all the occurrences of 'editor' by wrapping them up in brace
# brackets.
echo "My favorite programming editor is Emacs. Another editor I like is Vim." | sed -e s/editor/{&}/g
# My favorite programming {editor} is Emacs. Another {editor} I like is Vim.

sed -e 's/[tT]hy/{&}/g'  # replace input string of thy with {thy} ignore case

sed 's/[0-9]\+ /**** /g'  # mask first 3 groups of cc # with *, let last one go
# 1234 5678 9101 1234 ==> **** **** **** 1234

sed -e 's/the /this /1'
# replace first occurrence of 'the' with 'this' case sensitive

sed -e 's/[tT][hH][yY]/your/g'
# For each line in a given input file, transform all the occurrences of the
#  word 'thy' with 'your'. The search should be case insensitive, i.e.
# 'thy', 'Thy', 'tHy' etc. should be transformed to 'your'.



# =============================================================================
# %% Linux Shell - Grep Sed Awk
# https://tldp.org/LDP/abs/html/textproc.html
# https://www.thegeekstuff.com/2009/03/15-practical-unix-grep-command-examples/
# https://tldp.org/LDP/Bash-Beginners-Guide/html/sect_04_02.html
# https://www.gnu.org/software/sed/manual/html_node/Regular-Expressions.html

# grep A
# print only lines with any of the 4 words: the, that, then, those
grep -Eiw 'th(e|ose|en|at)'

# print out lines with consecutive repeated digits, can have space between
grep -E '(1 1|11|2 2|33|3 3|44|4 4|55|5 5|66|6 6|77|7 7|88|8 8|99|9 9|00|0 0)'
# grep B
















































# =============================================================================
