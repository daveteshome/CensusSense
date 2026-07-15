"""Static state name/abbreviation/FIPS reference. Hardcoded because it
never changes and doesn't need a Snowflake round trip to obtain."""

STATE_ABBREV_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}

STATE_NAME_TO_ABBREV = {name.lower(): abbrev for abbrev, name in STATE_ABBREV_TO_NAME.items()}

# The live 2020_METADATA_CBG_FIPS_CODES table has a NULL `STATE` abbreviation
# for these territories (only STATE_FIPS + COUNTY are populated). FIPS codes
# per the Census Bureau's standard state/territory code list.
TERRITORY_FIPS_TO_NAME = {
    "60": "American Samoa",
    "66": "Guam",
    "69": "Northern Mariana Islands",
    "72": "Puerto Rico",
    "78": "U.S. Virgin Islands",
}
