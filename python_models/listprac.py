foods = []
prices = []
sizes = ["big", "medium", "small"]
order_sizes = []
total = 0

while True:
    food = input("Enter food: ")
    if food.lower() == "q":
        break
    else:
        price = float(input("Enter price: "))
        foods.append(food)
        prices.append(price)
        size = input("Enter size: ").lower()
        if size in sizes:
            order_sizes.append(size)
            print("order confirm")

print("your cart")

for i in range(len(foods)):
    print(f"Food: {foods[i]}, Size: {order_sizes[i]}, Price: {prices[i]}")

total = sum(prices)
print(f"Total: {total}")




