from dataclasses import dataclass

def valid_hex_string(input):
    return (input.startswith("#") and len(input) == 7 
            and set(input[1:]).issubset({*map(str, range(10)), "a", "b", "c", "d", "e", "f"}))

@dataclass
class ColorPalettes:
    yellow_purple_black = ["#E0D12B", "#DAA520", "#FFBF00", "#FF9500", "#FFA500", "#FF7F50",
        "#FA8072", "#F08080", "#FFB6C1", "#E0BE4C", "#F5CAAF", "#DA70D6", "#BA55D3", "#9370DB",
        "#8A2BE2", "#6A0DAD", "#4B0082", "#2E0854", "#1A0028", "#000000"]

    black_red_blue = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#FF7F50",
        "#B8860B", "#FFC700", "#bdb76b", "#6b8e23", "#556b2f", "#b3cde0", "#a1c4d6",
        "#8bbdd9", "#7aaed4", "#699ecf", "#5e94c9", "#4b82b4"]    

    purple_red_blue = ["#411d31", "#631b34", "#32535f", "#0b8a8f", "#0eaf9b", "#30e1b9"]

    black_magenta_purple = ["#000000", "#b80049", "#ea569e", "#ffa653", "#fbe7b5", "#ff89dc",
        "#bb19e1", "#4a17a1", "#071c5a"]

    red_blue_red = ["#401b20", "#8e252e", "#9350aa", "#0e3abf", "#24793d", "#ffab89",
        "#fc4e51", "#de024e"]

    black_purple = ["#130208", "#1f0510", "#31051e", "#460e2b", "#7c183c", "#d53c6a",
        "#ff8274"]

    black_orange_yellow = ["#202215", "#3a2802", "#963c3c", "#ca5a2e", "#ff7831",
        "#f39949", "#ebc275", "#dfd785"]

    blue_gray_pink = ["#292831", "#333f58", "#4a7a96", "#ee8695", "#fbbbad"]

    red_blue_black = ["#de024e", "#fc4e51", "#ffab89", "#24793d", "#0e3abf",
        "#9350aa", "#8e252e", "#401b20"]

    red_yellow_blue = ["#ee4035", "#f37736", "#fdf498", "#7bc043", "#0392cf", "#8409da"]

    black_green_orange = ["#000000", "#003300", "#006600", "#CC6600", "#993300"]

    red_orange_yellow = ["#660000", "#990000", "#CC3333", "#FF9900", "#FFC333", "#CCFFCC"]

    orange_blue_black = ["#9c2a0b", "#ab2d0a", "#bd350b", "#bd400b", "#bd4f0b", "#cc760c",
        "#cc7f0c", "#d9910b", "#d9ab16", "#4ab80f", "#0f61b8", "#0a2ba3", "#09188f", 
        "#081680", "#060f57", "#020733", "#00010a"]
    
    black_red_green = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000","#FF7F50",
        "#B8860B", "#FFC700", "#bdb76b", "#6b8e23", "#556b2f", "#006600", "#004d00"]

