def split_file(input_file, output_prefix, lines_per_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    num_files = (len(lines) + lines_per_file - 1) // lines_per_file

    for i in range(num_files):
        start = i * lines_per_file
        end = min((i + 1) * lines_per_file, len(lines))
        output_file = f"{output_prefix}_{i + 1:02}.re"
        with open(output_file, 'w') as f_out:
            f_out.writelines(lines[start:end])


input_file = "practical_regex/practical_regexes.re"
output_prefix = "practical_regex/practical_regexes"
lines_per_file = 10_000

split_file(input_file, output_prefix, lines_per_file)
