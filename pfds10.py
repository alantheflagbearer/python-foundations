#Create a dictionary for yourself. Access values, add a new key, update an existing one, then loop through and print everything neatly.

contact = {
    "name": "Mutasim Billah",
    "phone": "+1 929 661 1822",
    "city": "Brooklyn",
    "degree": "M.S. Computer Science"
}

print(f"name: {contact['name']}")
contact["email"] = "mutasim.billah@example.com"
contact["phone"] = "+1 929 661 1823"

for key, value in contact.items():
    print(f"{key}: {value}")
