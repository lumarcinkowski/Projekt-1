
using Pkg
Pkg.add("Statistics")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("CUDA")
using CUDA
using Statistics
using CSV
using DataFrames

# Przykładowe dane treningowe (ham i spam), odczytanie z pliku CSV

data = CSV.read("C:\\Users\\lmarc\\OneDrive\\Pulpit\\WNO\\Projekt 1\\emails.csv", DataFrame)
ham = filter(row -> row[:Category] == "ham", data)
spam = filter(row -> row[:Category] == "spam", data)

ham_messages = collect(ham[:, 2])

size_ham_messages = size(ham_messages, 1)
spam_messages = collect(spam[:, 2])
size_spam_messages = size(spam_messages, 1)

# Funkcja trenująca filtr Bayesa
function train_bayesian_filter(ham_messages, spam_messages)
    ham_word_counts = Dict{String, Int}()
    spam_word_counts = Dict{String, Int}()
    
    for message in ham_messages
        words = split(lowercase(message))
        for word in words
            ham_word_counts[word] = get(ham_word_counts, word, 0) + 1
        end
    end
    
    for message in spam_messages
        words = split(lowercase(message))
        for word in words
            spam_word_counts[word] = get(spam_word_counts, word, 0) + 1
        end
    end
    
    return ham_word_counts, spam_word_counts
end

# Funkcja klasyfikująca wiadomość jako ham lub spam
function classify_bayesian_filter(message, ham_word_counts, spam_word_counts,size_ham_messages, size_spam_messages)
    words = split(lowercase(message))
    ham_prob = size_ham_messages/(size_ham_messages+size_spam_messages)
    spam_prob = 1 - ham_prob
    
    elements_in_message_count = length(words)
    a = 1.0
    
    for word in words
        ham_prob *= (a+get(ham_word_counts, word, 0)) / (sum(values(ham_word_counts))+a*elements_in_message_count)
        spam_prob *= (a+get(spam_word_counts, word, 0)) / (sum(values(spam_word_counts))+a*elements_in_message_count)       
    end

    println("No spam probability:    ", ham_prob)  
    println("Spam probability:    ", spam_prob) 
    
    
    if ham_prob >= spam_prob
        return "This email is ham"
    
    else 
        return "This email is spam"
    end

    
end
# Przykład użycia
ham_word_counts, spam_word_counts = train_bayesian_filter(ham_messages, spam_messages)
test_message = "Free IPhone 15. Buy it now. 80% discount. "
classification = classify_bayesian_filter(test_message, ham_word_counts, spam_word_counts,size_ham_messages, size_spam_messages)

println(classification)
