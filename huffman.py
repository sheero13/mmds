import heapq
from collections import Counter

def huffman_coding_simplified(message):
    if not message:
        return {}, {}, ""

    frequency = Counter(message)
    
    heap = [[freq, [char, ""]] for char, freq in frequency.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        char = heap[0][1][0]
        return {char: 1}, {char: "0"}, "0" * frequency[char]

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        merged = [left[0] + right[0]] + left[1:] + right[1:]
        heapq.heappush(heap, merged)

    huffman_tree = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    huffman_codes = {char: code for char, code in huffman_tree}
    encoded_string = "".join(huffman_codes[char] for char in message)
    
    return frequency, huffman_codes, encoded_string

if __name__ == "__main__":
    input_message = input("Enter a message to compress: ")
    frequency, huffman_codes, encoded_string = huffman_coding_simplified(input_message)
    
    print("\n--- Huffman Coding Results ---")
    print(f"{'Character':<10} | {'Frequency':<10} | {'Huffman Code':<15}")
    print("-" * 45)
    for char, freq in sorted(frequency.items()):
        print(f"{char:<10} | {freq:<10} | {huffman_codes.get(char, ''):<15}")
    
    original_bits = len(input_message) * 8
    compressed_bits = len(encoded_string)
    
    print(f"\nEncoded Binary String:\n{encoded_string}")
    print(f"\nOriginal size:      {original_bits} bits")
    print(f"Compressed size:    {compressed_bits} bits")
    if compressed_bits > 0:
        print(f"Compression Ratio: {(original_bits / compressed_bits):.2f}")
