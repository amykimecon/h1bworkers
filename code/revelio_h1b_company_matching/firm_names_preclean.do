* This do file cleans firm names in Stata

********************************************************************************

ssc install distinct, replace

********************************************************************************

* Load in firm names
use "firm_names.dta", clear
// ** the user can upload the appropriate firm names as firm_names.dta, the variable for firm names should be "employer_original"

des

summarize N

di "number of unique employer originals"
distinct employer_original

********************************************************************************

* Create a variable for the cleaned names to make changes to
generate emp = employer_original

********************************************************************************

*Standardize punctuation and suffixes, prefixes
replace emp = subinstr(emp,"Incorporated","",.)
replace emp = substr(emp,1,length(emp)-4) if substr(emp, -4, 4)==" Inc"
replace emp = subinstr(emp,"&","And",.)
replace emp = subinstr(emp,"And Co.","",.)
replace emp = subinstr(emp,"Co.","",.)
replace emp = subinstr(emp,",","",.)
replace emp = subinstr(emp,".","",.)
replace emp = subinstr(emp,"/","",.)
replace emp = subinstr(emp," Llc","",.) if substr(emp, -4, 4)==" Llc"
replace emp = subinstr(emp," LLC","",.) if substr(emp, -4, 4)==" LLC"
replace emp = subinstr(emp," L L C","",.) if substr(emp, -6, 6)==" L L C"
replace emp = subinstr(emp,"'","",.)
replace emp = subinstr(emp,"-","",.)
replace emp = subinstr(emp," Limited","",.) if substr(emp, -8, 8)==" Limited"
replace emp = subinstr(emp," Ltd","",.) if substr(emp, -4, 4)==" Ltd"
replace emp = subinstr(emp,"Cntr","Center",.)
replace emp = subinstr(emp,"Company","",.)
replace emp = subinstr(emp,"Companies","",.)
replace emp = subinstr(emp,"The ","",.)
replace emp = subinstr(emp,"Corporation","",.)
replace emp = subinstr(emp,"Stores","",.)
replace emp = subinstr(emp,"Store","",.)
replace emp = subinstr(emp," Llp","",.) if substr(emp, -4, 4)==" Llp"
replace emp = subinstr(emp," Lp","",.) if substr(emp, -3,3)==" Lp"
replace emp = substr(emp,1,length(emp)-4) if substr(emp, -4, 4)==" Plc"
replace emp = substr(emp,1,length(emp)-4) if substr(emp, -4, 4)==" Org"
replace emp = subinstr(emp,"Theatre","Theater",.)
replace emp = substr(emp,1,length(emp)-4) if substr(emp, -4, 4)==" Com"

*Backslashes and the characters after them
forvalues i = 1/130 {
	replace emp = regexr(emp, "\\.", "")
}

*Specific replacements
replace emp = "FedEx" if ustrregexm(emp, "Federal Express")
replace emp = "United States Postal Service" if ustrregexm(emp, "USPS")
replace emp = "24 Hour Fitness" if strpos(emp,"24 Hour Fitness")>0
replace emp = "A Wireless Incorporated" if emp=="A Wireless Inorporated"
replace emp = "Abco Hvacr Supply Solutions" if emp =="Abcohvacrsupplysolutions"
replace emp = "Abec" if emp=="Abec Incorporated"
replace emp ="Abilene Regional Medical" if emp=="Abilene Regional Medical Center"
replace emp = "Absher Construction Company" if emp=="Absherconstructionco"
replace emp = "Adventist Health" if strpos(emp,"Adventist Medical Center")>0
replace emp = "Adventist Healthcare" if strpos(emp,"Adventist Healthcare")>0
replace emp = "Ball Corporation" if emp=="Ball Aerospace Technologies"
replace emp = "Ball Corporation" if emp=="Ball Aerospace"
replace emp = "Ball Corporation" if strpos(emp,"Ball Corporation")>0
replace emp = "Bank Of America" if strpos(emp,"Bank Of America")>0
replace emp = "Banner Health" if emp=="Banner Behavioral Health Hospital"|employer=="Banner Churchill Community Hospital"|employer=="Banner Estrella Medical Center"|employer=="Banner Health System"
replace emp="Bartell Drugs" if emp=="Bartell Drug Company"
replace emp="Bassett Healthcare" if emp=="Bassett Healthcare Network"
replace emp="Batteries Plus" if strpos(emp,"Batteries Plus")>0
replace emp="Baycare" if strpos(emp,"Baycare")==1
replace emp="Bayfront Health" if strpos(emp,"Bayfront Health")==1
replace emp="Bayfront Health" if emp=="Bayfront Medical Center"
replace emp="Bayhealth" if emp=="Bayhealth Medical Center"
replace emp="Beaumont Health System" if emp=="Beaumont Hospitals"
replace emp="Beebe Healthcare" if emp=="Beebe Medical Center"
replace emp="Belkin" if strpos(emp,"Belkin")==1
replace emp="Bellagio" if strpos(emp,"Bellagio")==1
replace emp="Belle Tire" if strpos(emp,"Belle Tire")==1
replace emp="Berkshire Hathaway Homeservices" if strpos(emp,"Berkshire Hathaway Homeservices")==1
replace emp="Bingham Memorial" if emp=="Bingham Memorial Hospital"
replace emp="Blain Supply" if emp=="Blain Supply Company"
replace emp ="Blue Cross Blue Shield North Dakota" if emp=="Blue Cross Blue Shield Nd"
replace emp ="Blue Cross Blue Shield Of Minnesota" if emp=="Blue Cross and Blue Shield of Minnesota"
replace emp ="Blue Cross Blue Shield Of Minnesota" if emp=="Bluecross Blueshield Minnesota"
replace emp="Blue Cross Blue Shield Of North Carolina" if emp=="Blue Cross and Blue Shield of North Carolina"|employer=="Bluecross Blueshield Of North Carolina"
replace emp="Blue Cross Blue Shield of Massachusetts" if emp=="Bluecross And Blueshield Of Massachusetts"
replace emp="Blue Cross Blue Shield Of Tennessee" if emp=="Bluecross Blueshield Tennessee"
replace emp="Bnsf Railway" if emp=="Bnsf"
replace emp="Bombardier" if emp=="Bombardier Aerospace"
replace emp="Bon Secours Health System " if emp=="Bon Secours Richmond Health"|employer=="Bon Secours St Francis"
replace emp="Bonner General Hospital" if emp=="Bonner General Health"
replace emp="Borgess Health" if emp=="Borgess"|employer=="Borgess Medical Center"
replace emp="Boyd Brothers Transportation" if emp=="Boyd Bros Transportation "|employer=="Boyd Bros"
replace emp="Bp America" if emp=="Bp Americas"
replace emp="Brockton Hospital" if emp=="Brockton Hospital Ma"
replace emp="Bronson Healthcare" if emp=="Bronson Healthcare Group"
replace emp="Brookdale Senior Living" if emp=="Brookdale"
replace emp="Brookhaven National Laboratory" if emp=="Brookhaven National Laboratories"
replace emp="Builders Transportation Company" if emp=="Builders Transportation"
replace emp = "Calsonic Kansei" if emp=="Calsonickansei North America"
replace emp="Capital Group" if strpos(emp,"Capital Group")>0
replace emp = "Caraco Pharmaceutical" if strpos(emp,"Caraco Pharmaceutical")>0
replace emp ="Caresouth" if strpos(emp,"Caresouth")==1
replace emp="Careworks Tech" if emp=="Careworks Technologies"
replace emp="Carilion Clinic" if emp=="Carilion"
replace emp="Carolinaeast Health System" if emp=="Carolinaeast Medical Center"
replace emp="Carquest" if emp=="Carquest Auto Parts"
replace emp="Cayuga Medical Center" if strpos(emp,"Cayuga ")==1
replace emp="Cenlar Fsb" if emp=="Cenlar"
replace emp="Centegra Health Systems" if emp=="Centegra Health System"
replace emp="Center For Human Development" if emp=="Center For Human Dev "
replace emp="Centra Health" if emp=="Centra Medical"|employer=="Centrahealth Com"
replace emp="Centracare" if emp=="Centracare Clinic"
replace emp="Central DuPage Health" if emp=="Central Dupage Hospital"
replace emp="Central Mississippi Medical Center" if emp=="Central Ms Medical Center Cmmc"
replace emp="Ch Robinson" if strpos(emp,"Ch Robinson")==1
replace emp="Chalk Mountain Services" if strpos(emp,"Chalk Mountain Services")==1
replace emp="Chegg" if emp=="Chegg Com"
replace emp="Chg Healthcare" if emp=="Chg Healthcare Services"
replace emp="Childrens Hospital Orange County" if strpos(emp,"Childrens Hospital Orange County")>0
replace emp="Childrens Hospital Los Angeles" if emp=="Childrens Hospital of Los Angeles"
replace emp="Citizens Memorial Hospital" if emp=="Citizens Memorial Healthcare"
replace emp="Clearlink" if emp=="Clearlink Technologies"
replace emp="Columbus Regional Health" if emp=="Columbus Regional Health Care"|employer=="Columbus Regional Hospital"
replace emp="Comfort Keepers" if strpos(emp,"Comfort Keepers")==1
replace emp="Compass Group" if strpos(emp,"Compass Group")==1
replace emp = "Aventura Hospital And Medical Center" if emp =="Aventura Hospital And Medical Cntr North Area"
replace emp = "ABM" if strpos(emp,"Abm ")==1
replace emp="ABM" if emp=="Abm"
replace emp = "Abrazo" if strpos(emp,"Abrazo ")==1
replace emp = "Acc" if strpos(emp,"Acc ")==1
replace emp = "Accelecare" if strpos(emp,"Accelecare ")==1
replace emp="Con Am" if strpos(emp,"Conam Management")==1
replace emp="Conde Nast" if emp=="Conde Nast Publications "
replace emp="Consolidated Edison" if strpos(emp,"Consolidated Edison")>0
replace emp="Consultis" if strpos(emp,"Consultis")==1
replace emp="Continental Automotive Systems" if emp=="Continental Automotive Systems USA"
replace emp="Conway Regional Health System" if emp=="Conway Regional Health Systems"
replace emp="Cook Systems International" if strpos(emp,"Cook Systems")==1
replace emp="Cooper University Hospital" if emp=="Cooper University Health Care"
replace emp="Cornerstone Healthcare" if emp=="Cornerstone Healthcare Group Holding"
replace emp="Cox Automotive" if emp=="Cox Auto Trader"
replace emp="Cox Health Systems" if emp=="Coxhealth"
replace emp="Cpm Group" if emp=="Cpm Group Us"
replace emp="Cps Security Solutions" if emp=="Cps Security Usa "
replace emp="Cree" if emp=="Cree Company"
replace emp="Crestron Electronics" if emp=="Crestron"
replace emp="Cricket Communications" if emp=="Cricket Wireless"
replace emp="Crown Automotive" if emp=="Crown Automotive Company"
replace emp="Crozer Keystone Health Systems" if emp=="CrozerKeystone Health System"
replace emp="Csg International" if emp=="Csg"
replace emp="Csx" if emp=="Csx Transportation"
replace emp="Cubic" if emp=="Cubic Applications "
replace emp="Cummins" if strpos(emp,"Cummins")==1
replace emp="Cushman And Wakefield" if emp=="Cushman And Wakefield National"
replace emp="CVS Health" if emp=="Cvs HealthMinute Clinic"|employer=="Cvs HealthProfessional"|employer=="Cvs Health2"
replace emp="Cyracom" if emp=="Cyracom International "
replace emp ="Daviess Community Hospital" if emp=="Daviess County Hospital"
replace emp = "Day Kimball Hospital" if emp=="Day Kimball Healthcare"
replace emp = "Daymon" if strpos(emp,"Daymon ")==1
replace emp="Dch Health System" if emp=="Dch Regional Medical Center"
replace emp="Deaconess Health System" if emp=="Deaconess Hospital"
replace emp ="Dealertrack " if emp=="Dealertrack Technologies"
replace emp="Dean Health System" if emp=="Dean Health Systems "
replace emp="Delphi" if emp=="Delphi Automotive"|employer=="Delphi Automotive Systems"
replace emp="Delray Medical" if emp=="Delray Medical Center"
replace emp="Dg Business Solutions" if emp=="Dgbs Dg Business Solutions"
replace emp="Diversicare" if strpos(emp,"Diversicare")==1
replace emp="Djo Global" if emp=="Djo"
replace emp="Dollar Bank" if emp=="Dollar Bank Fsb"
replace emp="Dreamworks Animation" if emp=="Dreamworks Animation L L C"
replace emp="Dunkin Donuts" if emp=="Dunkin Brands "
replace emp="Dura Automotive" if emp=="Dura Automotive Systems"
replace emp="Edelman" if emp=="Edelman Group"
replace emp="Einstein Noah Restaurant Group" if strpos(emp,"Einstein Noah Restaurant Group")>0
replace emp="Eisai" if emp=="Eisai  North America"
replace emp="Ej Gallo Winery" if emp=="Ej Gallo"
replace emp="Encore Capital Group " if emp=="Encore Capital "
replace emp="Ensign Group" if emp=="Ensign"|employer=="Ensign Therapy"
replace emp="Equinox" if emp=="Equinox Fitness Clubs"|employer=="Equinox Holdings Limited"
replace emp = "Erickson Living" if strpos(emp, "Erickson Living")>0
replace emp = "Ericsson" if emp=="Ericsson Eeu"
replace emp="Esf Camps" if emp=="Esf Summer Camps"
replace emp="Esterline" if strpos(emp,"Esterline")>0
replace emp="Eurest" if strpos(emp,"Eurest")==1
replace emp="Experian" if strpos(emp,"Experian")==1
replace emp="Ez Mart" if emp=="Ez Mart Stores "
replace emp="Fabergent" if emp=="Fabergent Consulting Services"
replace emp="Factset" if emp=="Factset Research Systems "
replace emp="Farmers Cooperative" if emp=="Farmers Cooperative Company"
replace emp="Farmland" if emp=="Farmland Foods "
replace emp="Fastaff Travel Nursing" if emp=="Fastaff"
replace emp="Faurecia" if emp=="Faurecia Automotive"
replace emp = "Fegs" if emp=="Fegs Health Human Services"|employer=="Fegs New York"
replace emp="First Citizens Bank" if emp=="First Citizens"
replace emp="First Financial Bank" if strpos(emp,"First Financial")>0
replace emp="First Horizon" if strpos(emp,"First Horizon National")>0
replace emp="First Midwest Bank" if emp=="First Midwest Bancorp"|employer=="First Midwest Bank America"|employer=="First Midwest Bank Usa"
replace emp="First National Bank" if strpos(emp,"First National Bank")==1
replace emp = "First National Community Bank" if emp=="First National Community Bancorp"
replace emp="First Niagara Financial Group" if emp=="First Niagara"
replace emp="First Pointe Management Group" if strpos(emp,"First Pointe Management Group")>0
replace emp="First Supply" if emp=="First Supply Group "
replace emp="Firstbank" if emp=="Firstbank Holding Company"
replace emp="Firstenergy" if emp=="Firstenergy Corporate And Support Services"
replace emp="Firstlight Homecare" if strpos(emp,"Firstlight Homecare")==1
replace emp="Firstservice Residential" if strpos(emp,"Firstservice Residential")==1
replace emp="Fitness Formula Clubs" if strpos(emp,"Fitness Formula")>0
replace emp="Five Star Automotive" if emp=="Five Star Automotive Group"
replace emp="Fleetmatics" if strpos(emp,"Fleetmatics")==1
replace emp="Flexion" if emp=="Flexion Therapeutics"
replace emp="Flexton" if emp=="Flexton In"
replace emp="Flik" if strpos(emp,"Flik")==1
replace emp="Flint Hills Resources" if emp=="Flint Hills Resources Lp"
replace emp="Floor Decor" if emp=="Floor Decor Outlets America"
replace emp="Florida Aventura Aventura Hospital" if strpos(emp,"Florida Aventura Aventura Hospital")>0
replace emp="Florida Fort Pierce Lawnwood Regional" if strpos(emp,"Florida Fort Pierce Lawnwood Regional")>0
replace emp="Florida Fort Walton Beach Fort Beach Medical Center" if strpos(emp,"Florida Fort Walton Beach Fort")>0
replace emp="Florida Hospital" if strpos(emp,"Florida Hospital ")==1
replace emp="Florida Jacksonville Memorial Hospital" if emp=="Florida Jacksonville Memorial Hospital Of"
replace emp="Florida Kissimmee Osceola Regional Medical Center" if emp=="Florida Kissimmee Osceola Regional"
replace emp="Florida Largo Largo Medical Center" if strpos(emp,"Florida Largo Largo Medical Center")>0
replace emp="Florida Ocala Ocala Regional Medical Center" if emp=="Florida Ocala Ocala Regional Medical"
replace emp="Florida Ocala West Community Hospital" if emp=="Florida Ocala West Community"
replace emp="Florida Orange Park Orange Park Medical" if emp=="Florida Orange Park Orange Park Medical Center"
replace emp="Florida Palatka Putnam Community Medical" if emp=="Florida Palatka Putnam Community Medical Center"
replace emp="Florida Plantation Plantation General" if strpos(emp,"Florida Plantation Plantation General")>0
replace emp="Florida Port Charlotte Fawcett Memorial Hospital" if emp=="Florida Port Charlotte Fawcett Memorial"
replace emp="Florida Power Light" if emp=="Florida Power Light Company"
replace emp="Florida Sanford Central Regional Hospital" if emp=="Florida Sanford Central Regional"
replace emp="Florida St Petersburg Palms Of Hospital" if emp=="Florida St Petersburg Palms Of"
replace emp="Florida Tallahassee Capital Regional Medical Center" if emp=="Florida Tallahassee Capital Regional"
replace emp="Florida Trinity Medical Center Of" if emp=="Florida Trinity Medical Center Of Trinit"
replace emp="Florida West Palm Beach Jfk Medical" if strpos(emp,"Florida West Palm Beach Jfk Medical")>0
replace emp="Fluor" if strpos(emp,"Fluor")==1
replace emp="Fluxx" if emp=="Fluxx Labs"
replace emp="Fmg" if strpos(emp,"Fmg ")==1
replace emp="Fogarty Finger " if emp=="Fogarty Finger Pllc"
replace emp="Foley Equipment" if strpos(emp,"Foley Equipment")==1
replace emp="Follett" if strpos(emp,"Follett ")==1
replace emp="Fort Collins Heating And Air Conditioning" if strpos(emp,"Fort Collins Heating And Ai")>0|employer=="Fortcollinsheatingairconditioning"
replace emp=subinstr(employer,"Orthopaedics","Orthopedics",.)
replace emp="Fortress Investment Advisors" if emp=="Fortress Investment"|employer=="Fortress Investment Group "
replace emp="Forward Air" if emp=="Forward Air Ltl"|employer=="Forward Air Solutions"|employer=="Forward Air Tqi"|employer=="Forward Air Transportation Services"
replace emp="Foster Wheeler" if strpos(emp,"Foster Wheeler")==1
replace emp="Foth" if emp=="Foth Production Solutions "
replace emp="Foulger Pratt" if emp=="Foulger Pratt Companies"
replace emp="Foundation Financial" if emp=="Foundation Financial Group"
replace emp = "Foundation Surgical Hospital" if strpos(emp,"Foundation Surgical Hospital")==1
replace emp ="Fountain Valley Regional Hospital" if emp=="Fountain Valley Regional Hospital And Medical Center"
replace emp="Four Seasons Environmental" if emp=="Four Seasons Environmenta"
replace emp="Four Seasons Hotels and Resorts" if strpos(emp,"Four Seasons Hotel")==1|strpos(emp,"Four Seasons Resort")==1
replace emp="Four Seasons Plumbing" if emp=="Fourseasonsplumbinginc"
replace emp="Foxwoods Resort Casino" if emp=="Foxwoods"
replace emp="Fpc" if strpos(emp,"Fpc ")==1
replace emp="Fragomen Del Rey Bernsen Loewy" if emp=="Fragomen Del Rey Bernsen And Loewy"
replace emp="Framestore" if emp=="Framestore Us"
replace emp="Franciscan Health System" if emp=="Franciscan Health Systems"
replace emp="Frank Group" if emp=="Frank Group International"
replace emp="Frankford Hospitals" if strpos(emp,"Frankford Hospital")==1
replace emp="Franklin Energy" if strpos(emp,"Franklin Energy Services")>0
replace emp="Franz Family Bakery" if emp=="Franz Bakery"
replace emp="Frasiers Plumbing And Heating" if emp=="Frasiersplumbingheatinginc"
replace emp="Frazee Paint" if emp=="Frazee Paint Wallcovering"
replace emp="Frazier And Deeter" if emp=="Frazier Deeter"
replace emp="Fred Beans" if strpos(emp,"Fred Beans")==1
replace emp="Fred Hutchinson Cancer Research Center" if emp=="Fred Hutchinson Cancer Re"|employer=="Fred Hutch"
replace emp="Fred Meyer" if emp=="Fred Meyer Stores"
replace emp="Freddys Frozen Custard And Steakburgers" if emp=="Freddys Frozen Custard"
replace emp="Freedom Consulting" if emp=="Freedom Consulting Group"
replace emp="Freedom Mortgage" if strpos(emp,"Freedom Mortgage")==1
replace emp="Freeman Empires" if emp=="Freeman EmpiresInc"
replace emp="French Gerleman" if emp=="French Gerleman Electric Company"
replace emp="French Lick Resort" if emp=="French Lick Springs Resort"
replace emp="Fresh Success Marketing" if emp=="Fresh Success Marketing Group"
replace emp="Frey Lutz" if emp=="Freylutzcorp"
replace emp="Frias Transportation Management" if emp=="Frias Transportation"
replace emp="Friendly Franchisees" if strpos(emp,"Friendly Franchisees")>0
replace emp="Ftd Companies" if emp=="Ftd Com"
replace emp="Fuddruckers" if strpos(emp,"Fuddruckers")==1
replace emp="Fugro" if strpos(emp,"Fugro ")==1
replace emp="Fujifilm Holdings" if emp=="Fujifilm Holdings America "
replace emp="Furniture Mart" if emp=="Furniture Mart Usa"
replace emp="G4S" if strpos(emp,"G4S")==1
replace emp = "Gables Residential" if emp=="Gables Residential Trust"
replace emp = "Gainsco Auto Insurance" if emp=="Gainsco "
replace emp="Ganley Auto Group" if emp=="Ganley Auto Stores"
replace emp="Gartner" if emp=="Gartner Group"
replace emp="Gate Petroleum" if emp=="Gate Petroleum Company"
replace emp="Gatehouse Media" if strpos(emp,"Gatehouse Media")==1
replace emp="Gdkn" if emp=="Gdkn Corp"
replace emp="Genesis Biotechnology Group" if emp=="Genesis Biotechnology"
replace emp="Genesis Health" if emp=="Genesis Health Group"
replace emp="Genesys" if emp=="Genesys Telecommunications Laboratories"|employer=="Genesys Infotech"
replace emp="Genex Services" if emp=="Genex"
replace emp="Genpact" if emp=="Genpact Jawood Healthcare Solutions"
replace emp="Gibsons Heating Plumbing" if emp=="Gibsons Heating And Plumbi" |employer=="Gibsonsheatingplumbinginc"
replace emp="Gilbarco" if emp=="Gilbarco Veeder Root"
replace emp="Gkn Aerospace" if emp=="Gkn"
replace emp="Glacier Bank" if emp=="Glacier Bancorp"
replace emp="Glc Associates" if emp=="Glc Group"
replace emp="Glendale Memorial Hospital And Health Center" if strpos(emp,"Glendale Memorial Hospital")==1
replace emp="Global Brands" if emp=="Global Brands Group Holding"
replace emp="Globalfoundries" if emp=="Globalfoundries U S"|employer=="Globalfoundries Us"
replace emp="Glory Global Solutions" if emp=="Glory Global Solutions International"
replace emp="Glumac" if emp=="Glumac International"
replace emp="Goldrich Kest Industries " if emp=="Goldrich And Kest Industries"|employer=="Goldrich And Kest Management Co"
replace emp="Goodman Manufacturing Company" if emp=="Goodman Manufacturing"
replace emp="Granite Construction Company" if emp=="Granite Construction"
replace emp="Greater Cincinnati Behavioral" if emp=="Greater Cincinnati Behavioral Health Services"
replace emp="Greatwide Logistics" if emp=="Greatwide Distribution Logistics"|employer=="Greatwide"|employer=="Greatwide Logistics Services"|employer=="Greatwide Truckload Management"
replace emp="Green Mountain Energy" if emp=="Green Mountain Energy Company"
replace emp="Greenville Health System" if emp=="Greenville Hospital System"
replace emp="Gresham Barlow" if emp=="Gresham Barlow 10J"
replace emp="Greystone Health Care Management" if emp=="Greystone Home Healthcare"
replace emp="Guardian Healthcare Providers" if strpos(emp,"Guardian Healthcare Providers")==1
replace emp="Guardian Life Insurance" if emp=="Guardian Life"
replace emp="Guardsmark" if emp=="Guardsmark Security"
replace emp="Guidepoint" if emp=="Guidepoint Global"
replace emp="Gulfstream Aerospace" if emp=="Gulfstream"
replace emp="Gundersen Lutheran" if emp=="Gundersen Lutheran Health"|employer=="Gundersen Health System"
replace emp="Guthrie Health" if strpos(emp,"Guthrie ")==1
replace emp="Hair Cuttery" if strpos(emp,"Hair Cuttery")>0
replace emp="Hal Smith Restaurants" if strpos(emp,"Hal Smith Restaurant")>0
replace emp="Haley Automotive" if strpos(emp,"Haley Auto")==1
replace emp="Halifax Health" if strpos(emp,"Halifax Health")==1
replace emp="Hallmark Aviation" if emp=="Hallmark Aviation Services"
replace emp="Hallmark Rehabilitation" if emp=="Hallmark Rehab"
replace emp="Halyard Health" if emp=="Halyard "
replace emp="Hannafords" if strpos(emp,"Hannaford")>0
replace emp="Hard Rock Hotel" if strpos(emp,"Hard Rock Hotel")>0
replace emp="Harlem Childrens Zone" if strpos(emp,"Harlem Childrens Zone")>0
replace emp="Harmons Grocery" if emp=="Harmons"
replace emp="Harmony Home Health Hospice" if emp=="Harmony Home Health And Hospice Care"|employer=="Harmony Home Healthcare"
replace emp="Harpercollins Publishers" if emp=="Harpercollins "
replace emp="Harris Caprock" if strpos(emp,"Harris Caprock")==1
replace emp="Harry And David" if strpos(emp,"Harry And David")>0
replace emp="Hartford Healthcare" if strpos(emp,"Hartford Healthcare")>0
replace emp="Hartt Transportation" if emp=="Hartt Transportation Systems"
replace emp="Harvard Vanguard" if strpos(emp,"Harvard Vanguard")>0
replace emp="Harvest Consulting" if emp=="Harvest Consulting Group"
replace emp="Hatfield Quality Meats" if strpos(emp,"Hatfield Quality Meats")>0
replace emp="Hatstand" if emp=="Hatstand Us"
replace emp="Hays Medical Center" if emp=="Hays Medical Cente"
replace emp="Hb Fuller" if emp=="Hb Fuller Company"
replace emp="Hca" if strpos(emp,"Hca ")>0|employer=="Hcahealthcare"
replace emp="Hcl Technologies" if emp=="Hcl Technologies America "
replace emp="Hcr Home Care" if emp=="Hcr Come Care"
replace emp="Healthmarkets Insurance" if emp=="Healthmarkets Insurance Agency"
replace emp="Healthport" if emp=="Healthport Technologies "
replace emp="Healthsmart" if emp=="Healthsmart Holdings"
replace emp="Healthtrax" if strpos(emp,"Healthtrax")>0
replace emp="Healthworks Fitness Centers For Women" if emp=="Healthworks Fitness For Women"
replace emp="Heartland Automotive Services" if emp=="Heartland Automotive"
replace emp="Hei Hotels And Resorts" if emp=="Hei Hotels"
replace emp="Heico" if emp=="Heico Companies"
replace emp="Hektoen Institute" if emp=="Hektoen Institute L L C"
replace emp="Helix Energy Solutions Group" if emp=="Helix Esg"
replace emp="Hendrick Health System" if emp=="Hendrick Medical Center"
replace emp="Henry Mayo Newhall Memorial Hospital" if strpos(emp,"Henry Mayo Newhall")>0
replace emp="Heritage Oaks Bank" if emp=="Heritage Oaks Bancorp"
replace emp="Hersha Hospitality Management" if emp=="Hersha Hospitality Management Hhm"
replace emp="Hertrich Family of Automobile Dealers" if emp=="Hertrichs Family Of Automobile Dealership"
replace emp="Herzing University" if emp=="Herzing University System"
replace emp="Hhgregg Appliances" if emp=="Hh Gregg"|employer=="Hhgregg"
replace emp="Hibu" if emp=="Hibu Sales"
replace emp="High Point Regional Health System" if emp=="High Point Regional Health"
replace emp="Highlands Regional Medical Center" if emp=="Highlands Regional Medical CenterHma"
replace emp="Hill Physicians Medical Group" if strpos(emp,"Hill Physicians Medical Group")>0
replace emp="Hillcrest Healthcare System" if emp=="Hillcrest Health Systems"
replace emp="Hiller Plumbing Heating And Cooling Co" if emp=="Hiller Plumbing Heating"
replace emp="Hilti" if emp=="Hilti Usa"
replace emp="Hines Interests" if emp=="Hines Interests Partnership"
replace emp="Hms Host" if emp=="Hms Host International"
replace emp="Hms Holdings" if emp=="Hms Company"
replace emp="Holiday Inn" if emp=="Holiday Innclub Vacations"
replace emp="Hollywood Presbyterian Medical Center" if strpos(emp,"Hollywood Presbyterian")==1
replace emp="Holmes Murphy" if emp=="Holmes Murphy Associates"
replace emp="Holzer Health Systems" if emp=="Holzer Medical Center"
replace emp="Home Health Care Of Middle Tennessee" if emp=="Home Health Care Middle Tn"
replace emp="Home Helpers" if strpos(emp,"Home Helpers")==1
replace emp="Homepages" if emp=="Homepages Directories"
replace emp="Homes For Independent Living" if strpos(emp,"Homes For Independent Living")==1
replace emp="Homeserve" if emp=="Homeserve Usa"
replace emp="Honey Baked Ham Company" if emp=="Honey Baked Ham"|emp=="Honeybaked Ham"
replace emp="Honorhealth" if emp=="Honorhealth Medical Group"|emp=="Honor Health Network"
replace emp="Hooman Automotive Group" if emp=="Hooman Automotive"
replace emp="Hooters" if emp=="Hooters America "
replace emp="Horizon Pharma" if emp=="Horizon Pharma Usa"
replace emp="Horizontal Boring Tunneling" if emp=="Horizontalboringtuneling"
replace emp="Hornbeck Offshore Services" if emp=="Hornbeck Offshore"
replace emp="Horsburgh Scott Company" if emp=="Horsburgh Scott"
replace emp="Hospital Sisters Health Systems Hshs" if emp=="Hospital Sisters Health System"|emp=="Hospital Sisters Health"
replace emp="Hubbell Realty" if strpos(emp,"Hubbell Realty")==1
replace emp="Hudson Hospital And Clinics" if emp=="Hudson Hospital And Clinic"
replace emp="Huffines Auto Dealerships" if emp=="Huffines Auto Group"
replace emp="Hulu" if strpos(emp,"Hulu ")==1
replace emp="Humcap" if employer=="Humcap Lp"
replace emp="Hunter Engineering Company" if emp=="Hunter Engineering"
replace emp="Hunterdon Health Care System" if emp=="Hunterdon Health Care"|emp=="Hunterdon Medical Center"
replace emp="Huntington Learning Centers" if emp=="Huntington Learning Center"
replace emp="Hussmann" if strpos(emp,"Hussmann")==1
replace emp="Hutchinson Regional Healthcare System" if emp=="Hutchinson Regional Medical Center"
replace emp="Huxley Banking" if strpos(emp,"Huxley Banking")==1
replace emp="Hyde Engineering Consulting" if emp=="Hyde Engineering Plus Consulting"
replace emp="Hydrite Chemical Company" if strpos(employer,"Hydrite Chemical")>0
replace emp="Hydrochem Company" if strpos(emp,"Hydrochem")==1
replace emp="Hydrogen Group" if emp=="Hydrogen Uk"
replace emp="Hyundai Autoever" if strpos(emp,"Hyundai Autoever")==1
replace emp="Hyundai" if strpos(emp,"Hyundai")>0
replace emp="Icebergs One Hour Heating And Air Conditioning" if emp=="Icebergsonehourheatingairconditioning"
replace emp="Icf International" if emp=="Icf Macro"|emp=="Icf"|emp=="Icf "
replace emp="Icma Retirement" if strpos(emp,"Icma")==1
replace emp="Icw Group" if emp=="Icw Group Insurance Company Of The West"
replace emp="Idc Technologies" if strpos(emp,"Idc ")==1|emp=="Idc"
replace emp="Ideal Protein" if emp=="Ideal Protein Of America"
replace emp="Idg" if emp=="Idg Technetwork"|emp=="Idg Usa "
replace emp="Idirect" if emp=="Idirect Technologies"
replace emp="Ifs North America" if emp=="Ifs"
replace emp="Igate" if strpos(emp,"Igate ")>0
replace emp="Iheart Media" if emp=="Iheartmedia Entertainment And Media Industry"|emp=="Iheartradio"
replace emp="Iir Holdings" if emp=="Iir Usa"
replace emp="Imation" if emp=="Imation Corp Us"
replace emp="Imethods" if emp=="Imethods Hire Methods"
replace emp="Imi Material Handling Logistics" if emp=="Imi Logistics"
replace emp="Immunotek Bio Centers" if emp=="Immunotek Bio Center "
replace emp="Impac Mortgage Holdings" if emp=="Impac Mortgage "
replace emp="Imperial Beverage" if emp=="Imperial Beverage Company"
replace emp="Incoln" if emp=="Incoln Corp"
replace emp="Indecomm" if strpos(emp,"Indecomm")==1
replace emp="Indegene" if employer=="Indegene Ttm"
replace emp="Indiana Terre Haute Terre Haute Regional Hospital" if emp=="Indiana Terre Haute Terre Haute Regional"
replace emp="Indotronix" if strpos(emp,"Indotronix")==1
replace emp="Industrial Dynamics" if emp=="Industrial Dynamics Company"
replace emp="Ineos" if strpos(emp,"Ineos")==1
replace emp="Inficare Technologies" if strpos(emp,"Inficare")==1
replace emp="Inficon" if emp=="Inficon Holding Ag"
replace emp="Infineon Technologies" if strpos(emp,"Infineon Technologies")==1
replace emp="Infinity Insurance Company" if strpos(emp,"Infinity Insurance")==1
replace emp="Infolob Solutions" if emp=="Infolob"
replace emp="Informa" if strpos(emp,"Informa ")==1
replace emp="Information Builders" if emp=="Information Builders Uk"
replace emp="Infoscitex" if strpos(emp,"Infoscitex")==1
replace emp="Infotech Enterprises" if emp=="Infotech Enterprises America"
replace emp="Infovision Consultants" if emp=="Infovision Consultant"
replace emp="Infrasource" if strpos(emp,"Infrasource")==1
replace emp="Ingenesis Group" if emp=="Ingenesis "
replace emp="Ingram Content Group" if emp=="Ingram Content Group Icg"
replace emp="Ingredion" if strpos(emp,"Ingredion")==1
replace emp="Inland Empire Health Plan" if emp=="Inland Empire Health Plan Iehp"
replace emp="Inmotion Entertainment" if emp=="Inmotion Entertainment Group"
replace emp="Innerworkings" if emp=="InnerworkingsInc"
replace emp="Innomethods Corp" if emp=="Innomethods"
replace emp="Innominds Software" if emp=="Innominds"
replace emp="Inova" if emp=="Inova Health System"|emp=="Inova Fairfax Hospital"|emp=="Inova Diagnostics "
replace emp="Insigma" if emp=="Insigmainc Com"
replace emp="Insituform" if emp=="Insituform Technologies"
replace emp="Instrumentation Laboratory" if emp=="Instrumentation Laboratory Company"
replace emp="Integramed America" if emp=="Integramed"
replace emp="Integrated Medical Systems" if strpos(employer,"Integrated Medical Systems")==1
replace emp="Integrity Staffing Solutions" if emp=="Integrity Staffing"
replace emp="Interactive Business Systems" if emp=="Interactive Business Systems Ibs"
replace emp="Intercontinental Hotels Group" if emp=="Intercontinental Hotel Group"|emp=="Intercontinental Kansas City"|emp=="Intercontinental Kansas City At The Plaza"|emp=="Intercontinental New Orleans"|emp=="Intercontinental Saint Paul Riverfront"|emp=="Intercontinental Tampa"
replace emp="Interface Rehab" if emp=="Interface Rehabilitation"
replace emp="Interim Lsu Hospital" if emp=="Interim Lsu Hospital Umc No"
replace emp="International Auto Components" if strpos(emp,"International Automotive Components")==1
replace emp="International Cruise And Excursions" if emp=="International Cruise And Excursions Ice"
replace emp="International Data Group" if emp=="International Data Group Idg"
replace emp="International Medical Corps" if emp=="International Medical Corps Ca"
replace emp="International Millennium Consultants" if emp=="International Millennium Consultants Imc"
replace emp="Interpath Laboratory" if emp=="Interpath Laboratories"
replace emp="Interpublic Group" if emp=="Interpublic Group Of Cos"
replace emp="Interstate Hotels And Resorts" if strpos(emp,"Interstate Hotels And Resor")>0
replace emp="Intertek" if strpos(emp,"Intertek ")==1
replace emp="Intlfcstone" if emp=="Intlfcstoneinc"
replace emp="Intrepid Payment Processing" if emp=="Intrepid Payment Processi"
replace emp="Intuitive Research And Technology" if emp=="Intuitive Research And Technology Corpor"
replace emp="Invincea" if emp=="Invincea Labs"
replace emp="Invision" if emp=="Invision Com"
replace emp="Ion Geophysical" if emp=="Ion Geophysical Corp"
replace emp="Ipc Hospitalist" if emp=="Ipc Hospitalist Company"|emp=="Ipc Healthcare"
replace emp="Ipg Photonics" if emp=="Ipg Photonics Corp"
replace emp="Ips Integrated Project Services" if emp=="Ips A Integrated Project Services"
replace emp="Ipsen Biopharmaceuticals" if emp=="Ipsen Pharmaceuticals"
replace emp="Ipsos" if strpos(emp,"Ipsos ")==1
replace emp="Iqor" if strpos(emp,"Iqor ")==1
replace emp="Iqpc" if emp=="Iqpc Exchange"
replace emp="Iri" if strpos(emp,"Iri ")==1
replace emp="Iridium Communications" if strpos(emp,"Iridium Satellite")==1
replace emp="Isa Atx" if strpos(emp,"Isa Atx")==1
replace emp="Isabella Stewart Gardner Museum" if emp=="Isabella Stewart Gardner Museum Boston"
replace emp="Isagenix" if strpos(emp,"Isagenix ")==1
replace emp="Isi Commercial Refrigeration" if emp=="Isi Commercial Refrigeration 532"
replace emp="Isl Techsolutions" if emp=="Isl Techso"
replace emp="Isle Casino Racing Pompano Park" if strpos(emp,"Isle Casino Racing")==1
replace emp="Isleta Casino Resort" if emp=="Isleta Resort And Casino"
replace emp="Isoftstone" if emp=="Isoftstone Holdings"
replace emp="Isqft Construction Software" if emp=="Isqftconstructionsoftwaretechnologies"
replace emp="Itc Holdings" if emp=="Itc Holdings Corp"
replace emp="Itc Infotech" if emp=="Itc Infotech Usa "
replace emp="Itech Consulting Partners" if emp=="Itech Cons"
replace emp="Ito En North America" if emp=="Ito En Usa"
replace emp="Iu Health" if strpos(emp,"Iu Health ")==1
replace emp="Ivoclar Vivadent" if strpos(emp,"Ivoclar Vivadent")>0
replace emp="Jack Links" if strpos(emp,"Jack Links")==1
replace emp="Jack Williams Tire" if strpos(emp,"Jack Williams Tire")==1
replace emp="Jackpot Junction Casino Hotel" if strpos(emp,"Jackpot Junction")==1
replace emp="Jackson Nurse Professionals" if emp=="Jackson Nurse Professiona"
replace emp="Jackson Rancheria Casino Hotel" if emp=="Jackson Rancheria Casino Resort"
replace emp="Jackson State University" if emp=="Jackson State University Jsu"
replace emp="Jacob A Riis Neighborhood Settlement" if emp=="Jacob A Riis Neighborhood Settlement House"|emp=="Jacob Riis Neighborhood Settlement House"
replace emp="Jaguar Land Rover" if strpos(emp,"Jaguar")>0 & strpos(emp,"Land Rover")>0
replace emp="Jameson Health System" if emp=="Jameson Memorial Hospital"
replace emp="Jamisonsearch" if emp=="Jamison Search International"
replace emp="Jani King" if strpos(emp,"Jani King")==1
replace emp="Jani King" if strpos(emp,"JaniKing International")==1
replace emp="Jannus Group" if emp=="Jannus Formerly Mountain States Group"
replace emp="Janssen Pharmaceuticals" if emp=="Janssen Biotech"
replace emp="Jarrett Ford" if emp=="Jarrett Ford Automotive Group"
replace emp="Jasper Contractors" if emp=="Jasper Contractor"
replace emp="Javaji Systems Solutions" if emp=="Javaji System Solutions "
replace emp="Jb Consulting" if emp=="Jb Consulting Systems "
replace emp="Jbn Associates" if emp=="Jbn And Associates "
replace emp="Jc Penney" if emp=="Jc Penny"
replace emp="Jcpenney" if emp=="Jcpenny"
replace emp="Jd Block And Associates" if emp=="Jd Block Amp Associates "
replace emp="Jeff Wyler Automotive" if strpos(emp,"Jeff Wyler")==1
replace emp="Jennie Turkey Store" if emp=="Jennie Turkey Store Company"
replace emp="Jetblue" if emp=="Jetblue Airways "
replace emp="Jewel Osco" if emp=="Jewel Osco Pharmacy"
replace emp="Jfk Medical Center" if emp=="Jfk Medical Center North Campus"
replace emp="JoAnn Fabric" if strpos(emp,"Jo Ann Fabric")==1
replace emp="John Ascuagas Nugget Casino" if emp=="John Ascuagas Nugget"
replace emp="John C Lincoln Health Network" if strpos(emp,"John C Lincoln Med Center")>0|strpos(emp,"John C Lincoln Medical")>0
replace emp="John Lincoln Hospital" if emp=="John Lincoln Hospitals"
replace emp="John Muir Health" if emp=="John Muir Mt Diablo Health"
replace emp="Johns Hopkins Health System" if strpos(emp,"Johns Hopkins Health")==1
replace emp="Johnson Controls" if emp=="Johnson Control "
replace emp="Johnson Financial" if emp=="Johnson Financial Group"
replace emp="Johnson Matthey" if strpos(emp,"Johnson Matthey")==1
replace emp="Johnston Memorial Hospital" if emp=="Johnston Memorial Hospital Care"
replace emp="Johnstone Supply" if emp=="Johnstone Supply Nw"|strpos(emp,"Johnstonesupply")==1
replace emp="Joseph Michaels International" if emp=="Joseph Michaels "
replace emp="Jps Health Network" if strpos(emp,"Jps Physician")==1|strpos(emp,"Jps Health")==1
replace emp="Judge Rotenberg Center" if strpos(emp,"Judge Rotenberg")==1
replace emp="Juniper Networks" if strpos(emp,"Juniper Networks")==1
replace emp="Juno Therapeutics" if emp=="Juno Therapeautics"
replace emp="Jw Michaels" if emp=="Jw Michaels And Co"
replace emp="Kalahari Resorts" if strpos(emp,"Kalahari")>0 & strpos(emp,"Resort")>0
replace emp="Kalahari Resorts" if emp=="Kalahari Waterparks Reso"
replace emp="Kalispell Regional Healthcare" if emp=="Kalispell Regional Medical Center"
replace emp="Kaman" if strpos(emp,"Kaman ")==1
replace emp="Kanes Furniture" if strpos(emp,"Kanes Furniture ")==1
replace emp="Kao Usa" if emp=="Kao "
replace emp="Kaplan" if strpos(emp,"Kaplan Test")==1
replace emp="Karcher" if strpos(emp,"Karcher ")==1
replace emp="Karmanos Cancer Institute" if emp=="Karmanos Cancer Center Institute"
replace emp="Kate Spade" if emp=="Kate Spade New York"
replace emp="Kehe Distributors" if emp=="Kehe Distribution "
replace emp="Kellermeyer Bergensons" if strpos(emp,"Kellermeyer Bergensons")==1
replace emp="Kelly Moore Paints" if strpos(emp,"Kelly Moore")>0 & strpos(emp,"Paint")>0
replace emp="Kelly Tractor" if strpos(emp,"Kelly Tractor")>0
replace emp="Kelsey Seybold" if strpos(emp,"Kelsey Seybold")==1
replace emp="Kemin" if emp=="Kemin Industries "
replace emp="Ken Garff" if strpos(emp,"Ken Garff")==1
replace emp="Kenco Group" if emp=="Kenco"|emp=="Kenco Logistic Services"
replace emp="Kenworth Sales" if emp=="Kenworth Sales Company"
replace emp="Kern Medical Center" if emp=="Kern Health Systems"
replace emp="Keryx Biopharmaceuticals" if strpos(emp,"Keryx")==1
replace emp="Keste" if emp=="Keste Com"
replace emp="Kettering Health Network" if emp=="Kettering Health"
replace emp="Keurig" if strpos(emp,"Keurig")==1
replace emp="Keyence" if emp=="Keyence  America"
replace emp="Keystone Automotive" if strpos(emp,"Keystone Automotive")==1
replace emp="Kik Custom Products" if strpos(emp,"Kik Custom Products")>0|emp=="Kik Products "
replace emp="Kimball Electronics" if emp=="Kimball Electronics Group"
replace emp="Kimco Services" if strpos(emp,"Kimco Facility Services")>0
replace emp="Kindred Family Focus" if strpos(emp,"Kindred Family Focus")>0
replace emp="Kingman Regional Medical Center" if emp=="Kingman Regional"
replace emp="Kingston Healthcare Company" if emp=="Kingston Healthcare"
replace emp="Kjww Engineering" if strpos(emp,"Kjww Engineering")==1
replace emp="Knight Transportation" if strpos(emp,"Knight Refrigerated")==1|strpos(emp,"Knight Refrigered")==1|emp=="Knight Transportion"
replace emp="Knorr Bremse" if strpos(emp,"Knorr Bremse")>0
replace emp="Knowledgent" if strpos(emp,"Knowledgent")==1
replace emp="Korn Ferry" if strpos(emp,"Korn Ferry")==1|strpos(emp,"KornFerry")==1
replace emp="Kraus Anderson Construction" if strpos(emp,"Kraus Anderson Construction")>0
replace emp="Kreilkamp" if strpos(emp,"Kreilkamp ")==1
replace emp="Kum Go" if emp=="Kum And Go "
replace emp="Kurt Manufacturing" if strpos(emp,"Kurt Manufacturing")==1
replace emp="La Bella Baskets" if emp=="La Bella Baskets Com"
replace emp="Laboratory Sciences Of Az" if emp=="Laboratory Sciences Arizona"
replace emp="Laborworks" if emp=="Laborworks Industrial"
replace emp="Ladacin" if emp=="Ladacin Network"
replace emp="Lafarge" if strpos(emp,"Lafarge")==1
replace emp="Laitram" if emp=="Laitram L L C"
replace emp="Laka" if strpos(emp,"Laka And")==1
replace emp="Lakeland Bank" if emp=="Lakeland Bancorp "
replace emp="Lakeland Regional Health" if emp=="Lakeland Regional Medical Center"|emp=="Lakeland Medical Center"
replace emp="Lakes Region General Healthcare" if emp=="Lakes Region General Hospital"
replace emp="Lakeshore Medical Clinics" if emp=="Lakeshore Medical Clinic"
replace emp="Lamar Advertising" if emp=="Lamar Advertising Company"|emp=="Lamar Outdoor Advertising"
replace emp="Lancaster General Health" if emp=="Lancaster General Hospital"|strpos(emp,"Lancaster General Health")==1
replace emp="Landair" if strpos(emp,"Landair ")==1
replace emp="Langan Engineering And Environmental Services" if strpos(emp,"Langan")>0 & strpos(emp,"Engineering")>0 & strpos(emp,"Environmental")>0
replace emp="Langham Hotels International" if emp=="Langham Hospitality Group"|strpos(emp,"Langham Hotel")==1
replace emp="Lanyon" if emp=="Lanyon Company"|emp=="Lanyon Solutions"
replace emp="Laquinta Inn And Suites" if emp=="Laquinta Inns And Suites"
replace emp="Largo Medical Center" if emp=="Largo Medical Center Largo"
replace emp="Larry Miller Group" if emp=="Larry Miller"|emp=="Larry H Miller Group Of Companies"|emp=="Larry H Miller Automotive Division"
replace emp="Lasko Products" if emp=="Lasko Metal Products"
replace emp="Latrobe Hospital" if emp=="Latrobe Area Hospital "
replace emp="Launch Technical Workforce Solutions" if emp=="Launchtws"
replace emp="Lawnwood Regional Medical Center" if strpos(emp,"Lawnwood Regional Medical Center")==1
replace emp="Lawrence Berkeley National Lab" if strpos(emp,"Lawrence Berkeley National Lab")==1
replace emp="Lawrence Harvey" if emp=="Lawrence Harvey Enterprise"
replace emp="Lawrence Hospital" if emp=="Lawrence Hospital Center"
replace emp="Layne Christensen Company" if emp=="Layne Christensen"
replace emp="Layton Construction" if emp=="Layton Construction Company "
replace emp="Le Meridien" if strpos(emp,"Le Meridien ")==1
replace emp="Leake Watts" if strpos(emp,"Leake Watts")==1
replace emp="Learning Tree International" if emp=="Learning Tree"
replace emp="Leatherman" if emp=="Leatherman Tool Group "
replace emp="Leclerc Foods" if strpos(emp,"Leclerc Foods")==1
replace emp="Ledic Management Group" if emp=="Ledic"
replace emp="Lee Auto Malls" if strpos(emp,"Lee Auto Malls")==1
replace emp="Lee Memorial Health System" if emp=="Lee Memorial Group"
replace emp="Legoland" if strpos(emp,"Legoland")==1
replace emp="Legrand" if strpos(emp,"Legrand ")==1
replace emp="Leo Burnett Company" if emp=="Leo Burnett"
replace emp="Leprino Foods" if emp=="Leprino Foods Company"
replace emp="Les Schwab Tire Centers" if emp=="Les Schwab"
replace emp="Levy Restaurants" if emp=="Levy Restaurants Se"
replace emp="Lewan Associates" if emp=="Lewan"|emp=="Lewan Technology"
replace emp="Lewis Resources Management" if emp=="Lewis Resource Management "
replace emp="Lewisgale Medical Center" if strpos(emp,"Lewisgale Hospital")==1|emp=="Lewisgale Regional Health System"|emp=="Lewisgale Health System"
replace emp="Lgs Innovations" if emp=="Lgs Bell Labs Innovations"
replace emp="Lhoist Group" if emp=="Lhoist North America"
replace emp="Lhp Engineering Resource Division" if emp=="Lhp Engineerng Resources"
replace emp="Liberty Diversified" if strpos(emp,"Liberty Diversified ")==1
replace emp="Liberty Healthcare" if strpos(emp,"Liberty Healthcare")==1|strpos(emp,"Liberty Health Care")==1
replace emp="Liberty National Life Insurance" if strpos(emp,"Liberty National Life Insurance")==1|emp=="Liberty National Insurance Company"
replace emp="Liberty Regional Medical Center" if emp=="Liberty Region Medical Center"
replace emp="Licking Memorial Health Systems" if emp=="Licking Memorial"
replace emp="Liebherr" if strpos(emp,"Liebherr")==1
replace emp="Life Care Centers of America" if strpos(emp,"Life Care Center")==1
replace emp="Lifecare" if strpos(emp,"Lifecare ")==1
replace emp="Lifelinc" if strpos(emp,"Lifelinc")==1
replace emp="Lifestar Response" if strpos(emp,"Lifestar Response")==1
replace emp="Lifestyle Lift" if emp=="Lifestyle Lift Sic"
replace emp="Lifetime Health Medical Group" if emp=="Lifetime Health Medical Groups"
replace emp="Lifeworks" if emp=="Lifeworks Nw"
replace emp="Lighthouse Nursing Agency" if emp=="Lighthouse Nursing"
replace emp="Lima Memorial Health System" if emp=="Lima Memorial Hospital"
replace emp="Limbach" if strpos(emp,"Limbach")==1
replace emp="Lincoln Investment" if emp=="Lincoln Investment Group "|emp=="Lincoln Investment Planning"
replace emp="Lineage Logistics" if strpos(emp,"Lineage Logistics")==1
replace emp="Lithko" if emp=="Lithko Contracting "
replace emp="Littleton Regional Healthcare" if emp=="Littleton Regional Hospital"
replace emp="Loandepot" if emp=="Loandepot Com"
replace emp="Logitech" if emp=="Logitech International Sa"
replace emp="Long Realty Company" if emp=="Long Realty"
replace emp="Longview Fibre Company" if strpos(emp,"Longview Fibre")>0
replace emp="Longview International Technology Solutions" if emp=="Longview International"
replace emp="Loomis" if emp=="Loomis Ab"|strpos(emp,"Loomis Armored")>0|strpos(emp,"Loomis Fargo")>0
replace emp="Loon Mountain Resort" if emp=="Loon Mountain"
replace emp="Loram" if strpos(emp,"Loram Maintenance")>0
replace emp="Lord And Taylor" if emp=="Lord And Taylor  Hudsons Bay Company"
replace emp="Lorillard Tobacco Company" if emp=="Lorillard "
replace emp="Louis Dreyfus" if emp=="Louis Dreyfus Commodities "
replace emp="Louis Padnos Iron Metal" if strpos(emp,"Louis Padnos Iron Metal")>0
replace emp="Lovelace Medical Center" if emp=="Lovelace Medical Centers And Health Plan"
replace emp="Lovelace Respiratory Research Institute" if emp=="Lovelace Respiratory Research"
replace emp="Lowes" if emp=="Lowes Companies"
replace emp="Lowermybills" if emp=="Lowermybills Com"
replace emp="Loyola University Medical Center" if emp=="Loyola University Health System"|emp=="Loyola University Health System Us"
replace emp="Loyola University New Orleans" if emp=="Loyola University of New Orleans"
replace emp="Lpr Construction" if emp=="Lpr Construction Company"
replace emp="Lucasarts" if strpos(emp,"Lucasarts")>0
replace emp="Lucasfilm" if strpos(emp,"Lucasfilm")>0
replace emp="Lucky Brand Jeans" if emp=="Lucky Brand"
replace emp="Luk" if emp=="Luk Usa "
replace emp="Luxottica" if strpos(emp,"Luxottica")==1
replace emp="Lyondellbasell" if strpos(emp,"Lyondellbasell")==1
replace emp="Lyons Magnus" if emp=="Lyons Magnus Company"
replace emp="Macallister Machinery" if strpos(emp,"Macallister Machinery")>0
replace emp="Machinery Services Corp" if strpos(emp,"Machinery Services Corp")>0
replace emp="Macquarie" if strpos(emp,"Macquarie")==1
replace emp="Macys" if strpos(emp,"Macys ")==1
replace emp="Magnesita Refractories" if emp=="Magnesita Refractories Company"
replace emp="Magneti Marelli" if strpos(emp,"Magneti Marelli")>0
replace emp="Mahindra Engineering" if strpos(emp,"Mahindra Engineering")>0
replace emp="Mahle" if strpos(emp,"Mahle ")==1
replace emp="Mallinckrodt" if strpos(emp,"Mallinckrodt")==1
replace emp="Mandalay Bay" if strpos(emp,"Mandalay Bay")==1
replace emp="Mansfield Oil" if strpos(emp,"Mansfield Oil")==1
replace emp="Mapfre" if strpos(emp,"Mapfre ")==1
replace emp="Maquet" if emp=="Maquet Cardiovascular" | emp=="Maquet Gmbh And Co"
replace emp="Marcus Millichap" if strpos(emp,"Marcus Millichap")==1
replace emp="Markit" if emp=="Markit Group"|emp=="Markit Information Services"|emp=="Markit On Demand"
replace emp="Marshall Medical Center" if emp=="Marshall Medical Centers"
replace emp="Martin Brower Company" if emp=="Martin Brower"
replace emp="Martin Marietta" if strpos(emp,"Martin Marietta")>0
replace emp="Martin Transportation Systems" if emp=="Martin Transportation"
replace emp="Martinrea" if emp=="Martinrea International "
replace emp="Marvell Technology Group" if emp=="Marvell"
replace emp="Mary Black Health System" if emp=="Mary Black Health Systems"
replace emp="Mary Greeley Medical Center" if emp=="Mary Greely Medical Center"
replace emp="Mary Lanning Healthcare" if emp=="Mary Lanning Memorial Hospital"
replace emp="Mary Washington Healthcare" if emp=="Mary Washington Hospital"|emp=="Mary Washington Hospital Medicorp Health System"
replace emp="Mathematica" if emp=="Mathematica Policy Research"
replace emp="Mathis Brothers Furniture" if emp=="Mathis Brothers Furniture Company"
replace emp="Mathnasium" if strpos(emp,"Mathnasium")==1
replace emp="Maverik Country Store" if strpos(emp,"Maverik Country Store")==1
replace emp="Max Ermas Restaurants" if emp=="Max Ermas"
replace emp="Maxim Healthcare" if strpos(emp,"Maxim Healthcare")==1|strpos(emp,"Maxim Homecare")==1|emp=="Maxim Home Health Resources"|emp=="Maxim Travel Nurses"
replace emp="Mbi Energy Services" if emp=="Mbi Energy"
replace emp="Mccoys Building Supply" if strpos(emp,"Mccoys Building Supply")>0
replace emp="Mcfarland Clinic" if emp=="Mcfarland Clinic Pc"
replace emp="Mdstaffers" if emp=="Mdstaffers Com"
replace emp="Mead Johnson Nutrition" if emp=="Mead Johnson Nutritionals" | emp=="Mead Johnson And Company"
replace emp="Meadville Medical Center" if strpos(emp,"Meadville Medical Center")==1
replace emp="Mechoshade" if strpos(emp,"Mechoshade")==1
replace emp="Medexpress" if strpos(emp,"Medexpress")==1
replace emp="Mediavest" if strpos(emp,"Mediavest")==1
replace emp="Medical Transportation Management" if strpos(emp,"Medical Transportation Management")==1
replace emp="Medical University Sc" if emp=="Medical University of South Carolina"
replace emp="Medicus Healthcare Solutions" if emp=="Medicus"
replace emp="Medimpact Healthcare Systems" if emp=="Medimpact"
replace emp="Medstaff Healthcare Solutions" if emp=="Medstaff "
replace emp="Medstar Health" if emp=="Medstar Medical "|strpos(emp,"Medstar ")==1
replace emp="Menasha" if emp=="Menasha Packaging"
replace emp="Menlo Worldwide Logistics" if emp=="Menlo Ww Logistics"
replace emp="Merit Health" if strpos(emp,"Merit Health ")==1
replace emp="Meritage Homes" if strpos(emp,"Meritage Homes ")==1
replace emp="Meritor" if emp=="Meritor Automotive"
replace emp="Mersen" if emp=="Mersen Usa"
replace emp="Mesa Air Group" if emp=="Mesa Airlines"
replace emp="Mesirow Financial" if strpos(emp,"Mesirow Financial")==1
replace emp="Metro Goldwyn Mayer" if strpos(emp,"Metro Goldwyn Mayer")==1
replace emp="Miami Childrens Health System" if emp=="Miami Childrens Hospital"
replace emp="Mid America Apartment Commmunities" if emp=="Mid America Apartment Com"
replace emp="Mid Coast Parkview Health Services" if emp=="Mid Coast Parkview Health"
replace emp="Midas Auto Service" if strpos(emp,"Midas Auto Service")==1|strpos(emp,"Midas Auto Systems")==1
replace emp="Middlesex Health System" if emp=="Middlesex Hospital"
replace emp="Midsouth Bank" if emp=="Midsouth Bancorp "|emp=="Midsouth Bank Na"
replace emp="Miele" if emp=="Miele Appliances "|emp=="Miele Usa"
replace emp="Mikes Plumbing Heating" if emp=="Mikesplumbingheatingserviceinc"
replace emp="Miller Jones" if emp=="Miller Jones Company"
replace emp="Millers Ale House" if strpos(emp,"Millers Ale House")==1
replace emp="Milliman" if emp=="Milliman Usa "
replace emp="Milwaukee Center For Independence" if strpos(emp,"Milwaukee Center For Independence")==1
replace emp="Mimis Cafe" if emp=="Mimis Caf"
replace emp="Mintel" if emp=="Mintel Group"
replace emp="Mitratech" if emp=="Mitratech Holdings "
replace emp="Mizuho Bank" if emp=="Mizuho Corporate Bank"
replace emp="Mobius Industries" if emp=="Mobius Industries Usa"
replace emp="Moneygram" if strpos(emp,"Moneygram")==1
replace emp="Morgan Keegan" if emp=="Morgan Keegan Company"
replace emp="Morphotrust" if strpos(emp,"Morphotrust")==1
replace emp="Mott Macdonald" if strpos(emp,"Mott Macdonald")==1
replace emp="Mount Carmel Health System" if emp=="Mount Carmel Health"
replace emp="Mount Kisco Medical Group" if emp=="Mount Kisco Medical Group Pc"
replace emp="Mountain Area Health Education Center" if emp=="Mountain Area Health Education Center Mahec"
replace emp="Mpw Industrial" if strpos(emp,"Mpw Industrial ")==1
replace emp="Multiband" if emp=="Multiband USA"
replace emp="Munich Reinsurance America" if emp=="Munich Re"
replace emp="Murata Electronics" if emp=="Murata Electronics Na"|strpos(emp,"Murata Electronics")==1
replace emp="Mwa Fraternal Financial" if strpos(emp,"Mwa Fraternal Financial")==1
replace emp="Mylan" if emp=="Mylan Laboratories "
replace emp="Nabholz" if emp=="Nabholz Construction Serv"
replace emp="Nabors Industries" if strpos(emp,"Nabors ")==1
replace emp="Nalco" if emp=="Nalco Company"
replace emp="Napa Autocare" if emp=="Napa Autocare Centers"
replace emp="Naral Pro Choice" if strpos(emp,"Naral Pro Choice")==1
replace emp="Nasdaq Omx" if strpos(emp,"Nasdaq Omx")==1
replace emp="National Gypsum" if emp=="National Gypsum Company"
replace emp="National Rural Utilities Cooperative Finance" if emp=="National Rural Utilities Cooperative Finance  Nrucfc"
replace emp="Nautica" if emp=="Nautica Enterprises "
replace emp="Navarro Regional Hospital" if emp=="Navarro Regional"
replace emp="Navient" if emp=="Navient Solutions"
replace emp="Navigant Consulting" if emp=="Navigant Cymetrix"|emp=="Navigant Healthcare Cymetrix"|emp=="Navigant International"
replace emp="Nbh Bank" if emp=="Nbh Bank N A"
replace emp="Nbt Bank" if emp=="Nbt Bancorp "
replace emp="Ncci" if emp=="Ncci Holdings "
replace emp="Nelson Mullins" if strpos(emp,"Nelson Mullins")==1
replace emp="Nemacolin Woodlands Resort" if strpos(emp,"Nemacolin Woodlands Resort")==1
replace emp="Nesco" if strpos(emp,"Nesco ")==1
replace emp="Ness" if emp=="Ness Usa"
replace emp="Nestle Purina Petcare Company" if emp=="Nestl Ao Purina Petcare Company"|emp=="Nestld Purina Petcare Company"
replace emp="Nestle Usa" if emp=="Nestl Usa"
replace emp="Netbrain Technologies" if emp=="Netbrain"
replace emp="Netscout Systems" if emp=="Netscout"
replace emp="Neurocrine Biosciences" if emp=="Neurocrine"
replace emp="New York Air Brake" if emp=="New York Air Brake Company"
replace emp="New York Community Bank" if strpos(emp,"New York Community Bancorp")>0
replace emp="Newcrestimage" if emp=="Newcrestimage Management"
replace emp="Newt Global" if emp=="Newt Global Consulting"
replace emp="Nexeo Solutions" if emp=="Nexeo"
replace emp="Nexius" if emp=="Nexius Solutions"
replace emp="Nextcare" if emp=="Nextcare Holdings"|emp=="Nextcare Urgent Care"
replace emp="Nfi" if strpos(emp,"Nfi Industries")==1|strpos(emp,"Nfi Transportation")==1
replace emp="Ngk Spark Plugs" if emp=="Ngk Spark Plugs Usa "
replace emp="Nhc" if strpos(emp,"Nhc ")==1
replace emp="Nigel Frank" if strpos(emp,"Nigel Fran")==1
replace emp="Nine West" if emp=="Nine West Holdings"
replace emp="Nintendo of America" if strpos(emp,"Nintendo")==1
replace emp="Nixon Power Services" if emp=="Nixon Power Services Company"
replace emp="Nixon Uniform Service" if strpos(emp,"Nixon Uniform Service")==1
replace emp="Noahs Bagels" if emp=="Noahs New York Bagels"
replace emp="Nolan Real Estate" if emp=="Nolan Real Estate Services"
replace emp="Nomo Soho" if emp=="Nomo Soho Hotel"
replace emp="Nomura" if strpos(emp,"Nomura ")==1
replace emp="Noramtec Consultants" if emp=="Noramtec Consultant"
replace emp="Norcraft Companies" if emp=="Norcraft Company"
replace emp="Nortek" if emp=="Nortek Air Solutions"|emp=="Nortek Global Hvac"|emp=="Nortek Security And Control"
replace emp="North Florida Regional Medicalcenter" if emp=="North Florida Regional"
replace emp="North Fulton Regional Hospital" if emp=="North Fulton Hospital"
replace emp="North Highland" if emp=="North Highland Company"|strpos(emp,"North Highland Worldwide")==1
replace emp="North Mississippi Health Services" if emp=="North Mississippi Medical Center"
replace emp="North Shore Bank" if emp=="North Shore Bank Fsb"
replace emp="North Shore Mechanical" if emp=="North Shore Mechanical Contractors"
replace emp="North Slope Borough" if emp=="North Slope Borough Nsb"
replace emp="Northwell Health" if strpos(emp,"Northwell Health")==1
replace emp="Northwest Bank" if emp=="Northwest Bancorp "
replace emp="Northwest Protective" if emp=="Northwest Protective Service"
replace emp="Northwestern Counseling And Support Services" if emp=="Northwestern Counseling And Support ServicesInc"
replace emp="Northwood Realty Services" if emp=="Northwood Realty Services Ohio"
replace emp="Novamed" if emp=="Novamed Eyecare"
replace emp="Novant" if emp=="Novant Health"
replace emp="Novellus Systems" if emp=="Novellus"
replace emp="Novetta Solutions" if emp=="Novetta "|emp=="Novetta"
replace emp="Novozymes" if strpos(emp,"Novozymes")==1
replace emp="Novum Pharmaceutical" if strpos(emp,"Novum Pharmaceutical")==1
replace emp="Npc International" if strpos(emp,"Npc International")==1
replace emp="Npl Construction" if strpos(emp,"Npl Construction")==1
replace emp="Nta Life" if strpos(emp,"Nta Life ")==1
replace emp="Ntech Solutions" if emp=="Ntech Solution"
replace emp="Ntn Bearing" if strpos(emp,"Ntn Bearing ")==1
replace emp="Nuverra" if emp=="Nuverra Environmental Solutions"
replace emp="Nwp Services" if emp=="Nwp Service"
replace emp="Nxstage Medical" if emp=="Nxstage Kidney Care"
replace emp="Octapharma" if emp=="Octapharma Plasma"
replace emp="Ocwen Financial" if emp=="Ocwen"
replace emp="Oerlikon" if strpos(emp,"Oerlikon ")==1
replace emp="Ogletree Deakins" if strpos(emp,"Ogletree Deakins ")==1
replace emp="Olgoonik" if strpos(emp,"Olgoonik")==1
replace emp="Omni Hotels" if strpos(emp,"Omni")>0 & strpos(emp,"Hotel")>0
replace emp="Omni Hotels" if strpos(emp,"Omni")>0 & strpos(emp,"Resort")>0
replace emp="Omnova" if emp=="Omnova Solutions"|emp=="Omnova Solutions "
replace emp="Omron" if emp=="Omron Automation And Safety"
replace emp="Oneamerica Financial Partners" if strpos(emp,"Oneamerica")==1
replace emp="Onesource Virtual" if emp=="Onesource Virtual Hr"
replace emp="Onewest Bank" if strpos(emp,"Onewest Bank")==1
replace emp="Opelousas General Health System" if emp=="Opelousas General Hospital"
replace emp="Opis Senior Services Group" if emp=="Opis Senior Services Grou"
replace emp="Oportun" if strpos(emp,"Oportun ")==1
replace emp="Orbitz" if emp=="Orbitz Worldwide"
replace emp="Organovo" if emp=="Organovo Holdings"
replace emp="Orica" if emp=="Orica Mining Services"
replace emp="Orrick Herrington Sutcliffe" if strpos(emp,"Orrick")>0 & strpos(emp,"Herrington")>0 & (strpos(emp,"Sutcliffe")>0|strpos(emp,"Suttcliffe")>0)
replace emp="Orthofix" if strpos(emp,"Orthofix")==1
replace emp="Orthopaedic And Spine Center Of The Rockies" if emp=="Orthopaedic Center Rockies"
replace emp="Orvis" if emp=="Orvis Company "
replace emp="Oswego Health" if emp=="Oswego Hospital"
replace emp="Outokumpu" if strpos(emp,"Outokumpu")==1
replace emp="Outstanding Plumbing And Heating" if emp=="Outstanding Plumbing And He"|emp=="Outstandingplumbingheatinginc"
replace emp="Overlake Hospital Medical Center" if emp=="Overlake Medical Center"
replace emp="Oxfam" if strpos(emp,"Oxfam ")==1
replace emp="Oxford Instruments" if emp=="Oxford Instruments America"
replace emp="Oxford University Press" if strpos(emp,"Oxford University Press")>0
replace emp="P Murphy Associates" if emp=="P Murphy"
replace emp="P2S Engineering" if emp=="P2Sengineeringinc"
replace emp="Pacific Gas Electric Company" if emp=="Pacific Gas and Electric Company"
replace emp="Pacific Shipping Trucking Company" if emp=="Pacific Shipping"
replace emp="Pacific Sunwear" if emp=="Pacific Sunwear California "
replace emp="Pacificsource Health Plans" if emp=="Pacificsource"
replace emp="Pactera" if emp=="Pactera Technology Na"
replace emp="Palmetto Health" if strpos(emp,"Palmetto Health")==1
replace emp="Palo Alto Veterans Institute For Research" if strpos(emp,"Palo Alto Veterans Institute For Research")==1
replace emp="Papa Ginos" if emp=="Papa Ginos Holdings"
replace emp="Par Pharmaceutical Company" if emp=="Par Pharmaceutical Companies"
replace emp="Parallel Hr" if emp=="Parallelhr Solutions"
replace emp="Park Nicollet Health Services" if strpos(emp,"Park Nicollet")>0 & strpos(emp,"Health")>0
replace emp="Parker Drilling Company" if emp=="Parker Drilling Management Services"
replace emp="Parma Community General Hospital" if emp=="Parma Community General"
replace emp="Pathways Community Behavioral Healthcare" if emp=="Pathways Community Behavioral HealthcareInc"
replace emp="Patrice And Associates" if strpos(emp,"Patrice And Associates")==1|emp=="Patrice Associates "
replace emp="Pediatrix" if emp=="Pediatrix Medical Group"
replace emp="Peets Coffee and Tea" if strpos(emp,"Peets Coffee")==1
replace emp="Pei Wei Asian Diner" if emp=="Pei Wei"
replace emp="Pen Bay Healthcare" if emp=="Pen Bay Medical Center"
replace emp="Penguin Random House" if strpos(emp,"Penguin Random House")>0
replace emp="Peninsula Oil And Propane" if emp=="Peninsula Oil Propane"
replace emp="Pennrose Management Company" if emp=="Pennrose Management"
replace emp="Pennymac Loan Services" if emp=="Pennymac"
replace emp="Penske Truck Leasing" if strpos(emp,"Penske Truck Leasing")==1
replace emp="Peoplefirst Homecare And Hospice" if emp=="Peoplefirst Homecare Hospice"
replace emp="Performance Contracting Group" if emp=="Performance Contracting "
replace emp="Perfumania" if strpos(emp,"Perfumania")==1
replace emp="Perkins Eastman" if strpos(emp,"Perkins Eastman")==1
replace emp="Permasteelisa" if strpos(emp,"Permasteelisa")==1
replace emp="Pernod Ricard" if strpos(emp,"Pernod Ricard")==1
replace emp="Phase One Consulting" if emp=="Phase One Consulting Group "
replace emp="Phelps Memorial Hospital Center" if emp=="Phelps Memorial Hospital"
replace emp="Philips Lighting" if strpos(emp,"Philips Lighting")==1|strpos(emp,"Philips Lumileds Lighting Company")==1
replace emp="Physician Affiliate Group Of New York" if emp=="Physician Affiliate Group Of Ny"
replace emp="Physiotherapy Associates" if emp=="Physiotherapy Associates Pa"
replace emp="Pictsweet Company" if emp=="Pictsweet"
replace emp="Pieology" if strpos(emp,"Pieology")==1
replace emp="Pilkington" if strpos(emp,"Pilkington ")==1
replace emp="Pinkerton" if strpos(emp,"Pinkerton Government Services")==1|emp=="Pinkerton Consulting And Investigations"
replace emp="Pinnacle Foods" if strpos(emp,"Pinnacle Foods ")>0
replace emp="Pivotal Payments" if emp=="Pivotal Payments Direct Corp"
replace emp="Pixar" if emp=="Pixar Animation Studios"
replace emp="Plastic Omnium" if strpos(emp,"Plastic Omnium Auto")==1
replace emp="Platinum Hrm" if emp=="Platinum Hr Management"
replace emp="Plattform Advertising" if emp=="Plattform"
replace emp="Pls Logistics Services" if emp=="Pls Logistics"
replace emp="Pmxindustries" if emp=="Pmxindustriesinc"
replace emp="Pocono Health System" if emp=="Pocono Medical Center"
replace emp="Polar Bear Air Conditioning" if emp=="Polar Bear Air Conditioni"|emp=="Polarbearairconditioni"
replace emp="Polaris Software Lab" if strpos(emp,"Polaris Software Lab")==1
replace emp="Polsinelli" if emp=="Polsinelli Pc"|emp=="Polsinelli Shughart"
replace emp="Potawatomi Hotel And Casino" if emp=="Potawatomi Bingo Casino"
replace emp="Pottstown Memorial Medical Center" if emp=="Pottstown Medical Center"
replace emp="Power Packaging" if emp=="Power Packaging An Exel Company"
replace emp="Powertek" if emp=="Powertek Corporate"
replace emp="Pratt And Whitney" if emp=="Pratt Amp Whitney"
replace emp="Precision Plumbing Heating Cooling" if emp=="Precision Plumbing Heating And Cooling"
replace emp="Precision Plumbing" if emp=="Precisionplumbinginc"
replace emp="Preferred Home Health Care And Nursing Services" if emp=="Preferred Home Health Care"
replace emp="Premier Health" if strpos(emp,"Premier Health Partners")==1
replace emp="Premier Home Health Care" if emp=="Premier Home Health Care Services"
replace emp="Presence Health" if emp=="Presence Health Network"
replace emp="Press Ganey Associates" if emp=="Press Ganey"
replace emp="Prgx Global" if emp=="Prgx"
replace emp="Prime Healthcare Services" if strpos(emp,"Prime Healthcare Services")==1
replace emp="Primeflight" if emp=="Primeflight Aviation Services"
replace emp="Primepay" if emp=="Primepay Global"
replace emp="Prmg" if emp=="Prmg Paramount Residential Mortgage Group"|emp=="Paramount Residential Mortgage Group"
replace emp="Pro Med Healthcare" if emp=="Pro Med Healthcare Services"
replace emp="Professional Placement" if emp=="Professional Placement Se"
replace emp="Progressive Nursing" if strpos(emp,"Progressive Nursing")==1
replace emp="Progrexion" if emp=="Progrexion Asg"
replace emp="Projectline Services" if emp=="Projectline"
replace emp="Prolim Global" if strpos(emp,"Prolim")==1
replace emp="Prometric" if emp=="Prometric Sme"
replace emp="Promise Hospital Group" if strpos(emp,"Promise Hospital")==1
replace emp="Prosource Wholesale Floorcoverings" if strpos(emp,"Prosource Wholesale Floorcoverings")==1|emp=="Prosource Wholesale"
replace emp="Protingent" if emp=="ProtingentTechlink Nw"
replace emp="Providence Health And Services" if strpos(emp,"Providence Health And Services")>0 | emp=="Providence Health And Servi"|emp=="Providence Health Network"|emp=="Providence Health"|employer=="Providence Medical Group"
replace emp="Providence Memorial Hospital" if emp=="Providence Memorial"
replace emp="Psav Presentation Services" if emp=="Psav"
replace emp="Public Health Management Corp" if emp=="Public Health Management "
replace emp="Public Storage" if emp=="Public Storage Us"
replace emp="Puma" if emp=="Puma North America "
replace emp="Purina" if emp=="Purina Petcare"
replace emp="Qatar Airways" if strpos(emp,"Qatar Airways")==1
replace emp="Qbe" if strpos(emp,"Qbe ")==1
replace emp="Qc Holdings" if strpos(emp,"Qc Holdings")==1
replace emp="Qinetiq" if strpos(emp,"Qinetiq")==1
replace emp="Quanex Building Products" if emp=="Quanex "
replace emp="Quantum Integrators" if emp=="Quantum Integrators Group "
replace emp="Quarles And Brady" if emp=="Quarles Brady"
replace emp="Quikrete Companies" if strpos(emp,"Quikrete")==1
replace emp="Rabobank" if emp=="Rabobank America"
replace emp="Raising Canes" if strpos(emp,"Raising Canes")==1|emp=="Raising Canea Cs"
replace emp="Rakuten" if strpos(emp,"Rakuten ")==1
replace emp="Ramboll Environ" if emp=="Ramboll Environ Us "
replace emp="Randstad" if emp=="Randstad Technologies"
replace emp="Rapid Response Monitoring Services" if emp=="Rapid Response Monitoring"
replace emp="Rapp Company" if emp=="Rapp Usa"
replace emp="Raymarine" if strpos(emp,"Raymarine ")==1
replace emp="Reading Hospital Medical Center" if emp=="Reading Hospital"
replace emp="Ready Pac Foods" if emp=="Readypac"
replace emp="Real Estate Agent Magazine" if emp=="Real Estate Agent Magazin"
replace emp="Realogy Franchise Group" if emp=="Realogy Holdings Corp"
replace emp="Red Roof Inns" if emp=="Red Roof Inn"
replace emp="Red Wing Shoe Company" if emp=="Red Wing Shoes"
replace emp="Redbox" if emp=="Redbox Automated Retail "
replace emp="Redfishtechnology" if emp=="Redfishtech Com"
replace emp="Redfynn Technologies" if emp=="Redfynn"
replace emp="Redners Warehouse Markets" if emp=="Redners Warehouse Market"
replace emp="Regency Hospital Company" if strpos(emp,"Regency Hospital")==1
replace emp="Regus" if emp=="Regus Us"
replace emp="Renaissance Consulting Group" if emp=="Renaissance Consulting Gr"
replace emp="Renal Ventures Management" if emp=="Renal Ventures"
replace emp="Rentokil" if strpos(emp,"Rentokil ")==1
replace emp="Rescare" if strpos(emp,"Rescare ")==1
replace emp="Resolvit" if strpos(emp,"Resolvit ")==1
replace emp="Resort Lifestyle Communities" if strpos(emp,"Resort Lifestyle Communities")==1
replace emp="Resorts World Casino New York" if strpos(emp,"Resorts World Casino New York")==1
replace emp="Responsys" if emp=="Responsys Com"
replace emp="Restore Therapy Services" if emp=="Restore Therapy"
replace emp="Reval" if emp=="Reval Com"
replace emp="Rexam" if emp=="Rexam Beverage Can Americas"
replace emp="Rexel" if strpos(emp,"Rexel ")==1
replace emp="Rexnord" if emp=="Rexnord Industries"
replace emp="Reynolds Smith Hill" if emp=="Reynolds Smith Hills "
replace emp="Rgis" if strpos(emp,"Rgis Inventory")==1
replace emp="Rh White Construction" if emp=="Rh White Construction Company"
replace emp="Rha Behavioral Health Services" if strpos(emp,"Rha Behavioral Health Services")==1
replace emp="Rhett Bigham" if strpos(emp,"Rhett Bigham")==1
replace emp="Rheem Manufacturing" if strpos(emp,"Rheem Manufacturing")==1
replace emp="Rick Case" if strpos(emp,"Rick Case ")==1
replace emp="Rideout Health" if emp=="Rideout Memorial Hospital"
replace emp="Rimkus Consulting" if strpos(emp,"Rimkus Consulting")==1
replace emp="Ring Container Technologies" if emp=="Ring Container Company"
replace emp="Ritz Carlton" if strpos(emp,"Ritz Carlton")==1
replace emp="Rj Corman Railroad" if strpos(emp,"Rj Corman Railroad")==1
replace emp="Robert Baird Company" if emp=="Robert Baird And Company"
replace emp="Robert Wood Johnson Hospital" if strpos(emp,"Robert Wood Johnson")==1 & strpos(emp,"Hospital")>0
replace emp="Robins Morton Group" if emp=="Robins Morton"
replace emp="Roc Search" if emp=="Roc Search Us"
replace emp="Rochester General Health System" if emp=="Rochester General"|emp=="Rochester General Hospital"|emp=="Rochester Regional Health Systems"|emp=="Rochester Regional Health"
replace emp="Rodenhiser Plumbing Heating" if strpos(emp,"Rodenhiser")>0 & strpos(emp,"lumbing")>0
replace emp="Roehl Transport" if strpos(emp,"Roehl")==1
replace emp="Roland Associates" if emp=="Roland And Asscociates"
replace emp="Rolls Royce" if strpos(emp,"Rolls Royce ")==1
replace emp="Ronnoco Coffee" if strpos(emp,"Ronnoco Coffee")==1
replace emp="Roseburg Forest Products" if strpos(emp,"Roseburg Forest Products")==1
replace emp="Rosemount Emerson Process Management" if strpos(emp,"Rosemount Emerson Process")==1
replace emp="Rothstein Kass" if strpos(emp,"Rothstein Kass")==1
replace emp="Roundys Supermarkets" if emp=="Roundys"
replace emp="Roush" if strpos(emp,"Roush ")==1
replace emp="Royal Canin" if strpos(emp,"Royal Canin")==1
replace emp="Rr Donnelley" if emp=="Rr Donnelly Sons" | emp=="Rr Donnely"
replace emp="Rsi Home Products" if strpos(emp,"Rsi Home Products")==1
replace emp="Rtkl" if strpos(emp,"Rtkl")==1
replace emp="Ruiz Foods" if emp=="Ruiz Food Products "
replace emp="Russ Darrow Group" if emp=="Russ Darrow Automotive Group"
replace emp="Ryan Companies" if strpos(emp,"Ryan Companies")==1
replace emp="Ryan Herco Flow Solutions" if strpos(emp,"Ryan Herco")==1
replace emp="Saatchi Saatchi" if strpos(emp,"Saatchi")>0
replace emp="Safelite AutoGlass" if emp=="Safelite Glass "|emp=="Safelite Group "
replace emp="Sage Hospitality" if emp=="Sage Hospitality Resources"
replace emp="Saia" if strpos(emp,"Saia ")==1
replace emp="Saks" if strpos(emp,"Saks ")==1
replace emp="Salamander Hotels And Resorts" if strpos(emp,"Salamander Resort")>0 | strpos(emp,"Salamander Hospitality")>0
replace emp="Samson Rope" if emp=="Samson Rope Technologies"
replace emp="San Angelo Community Medical Center" if emp=="San Angelo Community Medical Ctr"
replace emp="San Antonio Community Hospital" if emp=="San Antonio Community Hosp"
replace emp="San Diego Gas Electric" if strpos(emp,"San Diego Gas Electric")==1
replace emp="San Ramon Medical Center" if emp=="San Ramon Regional Medical Center"
replace emp="Sandoval Regional Medical Center" if strpos(emp,"Sandoval Regional Medical Center")==1
replace emp="Sandridge Energy" if emp=="Sandridgeenergyinc"
replace emp="Sandvik" if strpos(emp,"Sandvik ")==1
replace emp="Sanmina" if emp=="Sanmina SCI"
replace emp="Santander" if emp=="Santander Bank"
replace emp="Sapp Bros" if strpos(emp,"Sapp Bros")==1|strpos(emp,"Sapp Brothers")==1
replace emp="Sargent Aerospace And Defense" if emp=="Sargent Controls Aerospace"
replace emp="Sarnova" if emp=="Sarnova Hc "
replace emp="Sas Retail" if emp=="Sas Retail Services"
replace emp="Saudi Basic Industries" if strpos(emp,"Saudi Basic Industries")==1
replace emp="Savatree" if emp=="Savatree Savalawn"
replace emp="Savemart Supermarkets" if emp=="Savemart"
replace emp="Saviance Technologies" if emp=="Saviance"
replace emp="Sb Phillips" if emp=="Sb Phillips Company "
replace emp="Sba Communications" if strpos(emp,"Sba Communication")>0
replace emp="Sbm Management Services" if emp=="Sbm Management "
replace emp="Sbm Offshore" if strpos(emp,"Sbm Offshore")==1
replace emp="Schaeffer Manufacturing" if strpos(emp,"Schaeffer Manufacturing")==1|emp=="Schaeffer Mfg Oil And L"
replace emp="Schilli" if emp=="Schilli Transportation Services"
replace emp="Schneider National" if emp=="Schneider Nional"
replace emp="Schnitzer Steel" if strpos(emp,"Schnitzer Steel")==1
replace emp="Schooner Retirement Community" if emp=="Schooner Estates Retirement Community"
replace emp="Schott" if emp=="Schott North America "
replace emp="Schuff Steel" if strpos(emp,"Schuff Steel")==1
replace emp="Schulte Hospitality Group" if strpos(emp,"Schulte Hospitality")==1
replace emp="Schwans" if strpos(emp,"Schwans ")==1|strpos(emp,"Schwan S")==1|emp=="Schwan"|emp=="Schwan Food"
replace emp="Science Museum Minnesota" if emp=="Science Museum Of Mn"
replace emp="Scientific Drilling International" if emp=="Scientific Drilling"
replace emp="Scientific Games" if strpos(emp,"Scientific Games International")==1
replace emp="Scl Health" if emp=="Scl Health System"
replace emp="Sco Family Of Services" if strpos(emp,"Sco Family Of Services")==1
replace emp="Scripps Health" if emp=="Scripps Healthcare"|emp=="Scripps Home Healthcare"
replace emp="Scripps Networks Interactive" if emp=="Scripps Networks"
replace emp="Sdlc Partners" if strpos(emp,"Sdlc Partners")==1
replace emp="Seagate Technology" if emp=="Seagate"|emp=="Seagate Us "
replace emp="Seakr" if strpos(emp,"Seakr ")==1
replace emp="Sears" if strpos(emp,"Sears ")==1 & strpos(emp,"Kbr")==0
replace emp="Seasons Healthcare Management" if emp=="Seasons Healthcare Manage"
replace emp="Seattle Childrens" if emp=="Seattle Childrena Cs"
replace emp="Seattle Metropolitan Credit Union" if emp=="Seattle Metropolitan Cr Un"
replace emp="Seattle Times" if strpos(emp,"Seattle Times ")==1
replace emp="Seattles Union Gospel Mission" if emp=="Seattles Union Gospel Missio"
replace emp="Seaworld" if strpos(emp,"Seaworld ")==1
replace emp="Secura Insurance" if strpos(emp,"Secura Insurance ")==1
replace emp="Sedgwick" if emp=="Sedgwick Claims Management Services"
replace emp="Sees Candies" if emp=="Sees Candy Shops "
replace emp="Seiberlich Trane" if strpos(emp,"Seiberlich Trane")==1
replace emp="Select Family Of Companies" if emp=="Select Family Of Companie"
replace emp="Select Specialty Hospitals" if emp=="Select Specialty Hospital"
replace emp="Selectquote" if emp=="Selectquote Insurance Services"
replace emp="Seniorbridge" if emp=="Seniorbridge Family Companies"
replace emp="Sentara Healthcare" if emp=="Sentara Medical Group"
replace emp="Sentinel Technologies" if emp=="Sentinel Call Center Solutions"
replace emp="Sepco Services" if strpos(emp,"Sepco Services ")==1
replace emp="Sequel Youth And Family Services" if strpos(emp,"Sequel Youth And Family Services")==1
replace emp="Servesolid" if emp=="Servesolid Com"
replace emp="Servpro" if emp=="Servpro Industries "
replace emp="Seton Health" if strpos(emp,"Seton ")==1 & strpos(emp,"Health")>0
replace emp="Severn Trent" if strpos(emp,"Severn Trent ")==1
replace emp="Sgs Group" if strpos(emp,"Sgs Automotive")==1 | strpos(emp,"Sgs Consulting")==1 | strpos(emp,"Sgs North America")==1 | strpos(emp,"Sgs Petroleum Service")==1
replace emp="Shamrock Foods Company" if emp=="Shamrock Foods"
replace emp="Shands Healthcare" if strpos(emp,"Shands ")==1 & strpos(emp,"Regional Medical Center")>0
replace emp="Shannon Health" if strpos(emp,"Shannon Health System")==1 | strpos(emp,"Shannon Medical Center")==1
replace emp="Shazam" if strpos(emp,"Shazam Entertainment")==1
replace emp="Shc Osborn" if emp=="Shc Osborn Medical Center"
replace emp="Shc Shea" if emp=="Shc Shea Medical Center"
replace emp="Shc Thompson" if emp=="Shc Thompson Medical Center"
replace emp="Shearers Foods" if emp=="Shearers Food"|emp=="Shearers Snacks "
replace emp="Sheehy Auto" if strpos(emp,"Sheehy Auto")==1
replace emp="Shell Oil" if emp=="Shell Exploration And Production Company"|emp=="Shell International Exploration And Production"|emp=="Shell Global Solutions United States"|emp=="Shell Uk"
replace emp="Shelter Insurance" if emp=="Shelter Insurance Companies"|emp=="Shelter Mutual Insurance Company"
replace emp="Shi International Corp" if emp=="Shi International"
replace emp="Shimadzu" if strpos(emp,"Shimadzu")==1
replace emp="Shive Hattery" if strpos(emp,"Shive Hattery")==1
replace emp="Sidley Austin" if strpos(emp,"Sidley Austin ")==1
replace emp="Sierra Pacific Mortgage" if strpos(emp,"Sierra Pacific Mortgage")==1
replace emp="Sierra Providence East Hospital" if emp=="Sierra Providence Eastside Hospital"
replace emp="Sierra Vista Regional Health Center" if emp=="Sierra Vista Regional Medical Center"
replace emp="Sifto Canada" if strpos(emp,"Sifto Canada")
replace emp="Silicon Graphics" if strpos(emp,"Silicon Graphics ")==1
replace emp="Silver Airways" if emp=="Silver Airways Corp"
replace emp="Simpson Housing" if emp=="Simpson Housing Lllp"
replace emp="Sirius XM Radio" if strpos(emp,"Siriusxm")==1
replace emp="Sis Systems Integration Solutions" if emp=="Sis Systems "
replace emp="Six3 Systems" if strpos(emp,"Six3 Systems")==1
replace emp="Skadden Arps Slate Meagher" if strpos(emp,"Skadden Arps Slate Meagher")==1
replace emp="Sleep Train" if strpos(emp,"Sleep Train")
replace emp="Sleepys" if emp=="Sleepy S"
replace emp="Smc" if emp=="Smc  America"
replace emp="Smci" if emp=="Smci Glendale"
replace emp="Smile Brands" if strpos(emp,"Smile Brands")==1
replace emp="Smithfield Foods" if strpos(emp,"Smithfield Farmland")==1 |emp=="Smithfield Hog Production Division"|emp=="Smithfieldfarmland"
replace emp="Smiths Food Drug Centers" if emp=="Smiths Food Drug"
replace emp="Smokey Bones" if strpos(emp,"Smokey Bones ")==1
replace emp="SnydersLance" if strpos(emp,"Snyder")==1 & (strpos(emp,"Lance")>0|strpos(emp,"lance")>0)
replace emp="Sofa Mart" if strpos(emp,"Sofa Mart ")==1
replace emp="Sofitel" if strpos(emp,"Sofitel")==1
replace emp="Softlayer Technologies" if emp=="Softlayer"
replace emp="Softtek" if strpos(emp,"Softtek ")==1
replace emp="Software Galaxy Systems" if emp=="Software Galaxy System"
replace emp="Software Paradigms International" if strpos(emp,"Software Paradigms International")==1
replace emp="Sogeti" if strpos(emp,"Sogeti ")==1
replace emp="Solarwinds" if strpos(emp,"Solarwinds ")==1
replace emp="Solarworld" if strpos(emp,"Solarworld ")==1
replace emp="Soliant" if strpos(emp,"Soliant")==1
replace emp="Solomonedwards" if strpos(emp,"Solomonedwards")==1
replace emp="Solu Technology" if strpos(emp,"Solu Technology")==1
replace emp="Sonepar" if emp=="Sonepar Usa "
replace emp="Sonnenalp Of Vail" if strpos(emp,"Sonnenalp")==1 & strpos(emp,"Vail")>0
replace emp="Sonora Behavioral Health" if emp=="Sonora Behavioral Health Hospital"
replace emp="Sonora Quest" if strpos(emp,"Sonora Quest")==1
replace emp="Sonora Regional Medical Center" if strpos(emp,"Sonora Regional Medical Center")==1
replace emp="Souplantation" if strpos(emp,"Souplantation")==1
replace emp="Source Refrigeration" if strpos(emp,"Source Refrigeration ")==1
replace emp="Sourcehov" if strpos(emp,"Sourcehov")==1
replace emp="South Bay Mental Health" if emp=="South Bay Mental Health Center"
replace emp="South Nassau Communities Hospital" if strpos(emp,"South Nassau Communities Hospital")==1
replace emp="Southern Bank" if emp=="Southern Bancorp"
replace emp="Southern Regional Health System" if emp=="Southern Regional Health"|emp=="Southern Regional Medical Center"
replace emp="Southern States Automotive" if strpos(emp,"Southern States Automotive")==1
replace emp="Southern Wine And Spirits" if strpos(emp,"Southern Wine")==1
replace emp="Southland Industries" if emp=="Southland Industries Si"
replace emp="Southwest Behavioral Health Services" if emp=="Southwest Behavioral Health"|emp=="Southwest Behavioral And Health Services"
replace emp="Southwest General Health Center" if emp=="Southwest General Health"
replace emp="Southwest Washington Medical Center" if emp=="Southwest Washington Med Center"
replace emp="Sovereign Medical" if emp=="Sovereign Medical Group "
replace emp="Sparrow Health System" if emp=="Sparrow Specialty Hospital"
replace emp="Specialized Transportation Services" if emp=="Specialized Transportsion Services"
replace emp="Spirit Halloween" if strpos(emp,"Spirit Halloween")==1
replace emp="Spiritairlines" if emp=="Spiritairlinesinc"
replace emp="Spotify" if strpos(emp,"Spotify")==1
replace emp="Springer Science Business Media" if strpos(emp,"Springer Science And Business Media")==1
replace emp="Squire Patton Boggs" if strpos(emp,"Squire Patton Boggs")==1
replace emp="St Anthonys Medical Center" if emp=="St Anthonya Cs Medical Center"
replace emp="St Helena Hospital Clearlake" if emp=="St Helena Clearlake"
replace emp="St John Knits" if strpos(emp,"St John Knits")==1
replace emp="St Johns Community Services" if emp=="St Johna Cs Community Services"
replace emp="St Johns Mercy Medical Center" if emp=="St Johna Cs Mercy Medical Center"
replace emp="St Josephs Hopital And Medical Center" if emp=="St Josephs Hopital And Med Ctr"
replace emp="St Lukes Cataract Laser Institute" if emp=="St Lukes Cataract Laser Institut"
replace emp="St Lukes Magic Valley Medical Center" if emp=="St Lukes Magic Valley Reg Med Center"
replace emp="St Vincents Health System" if emp=="St Vincenta Cs Health System"
replace emp="Stewartenterprises" if emp=="Stewartenterprisesinc"
replace emp="Stone Brewing Co" if emp=="Stone Brewing Company"
replace emp="Stop Shop Supermarket" if strpos(emp,"Stop Shop Supermarket")>0|strpos(emp,"Stop And Shop Supermarket")>0
replace emp="Stout Systems" if strpos(emp,"Stout Systems")==1
replace emp="Strategic Employment Partners" if strpos(emp,"Strategic Employment Partners")==1
replace emp="Suez" if emp=="Suez Na"
replace emp="Summa Health System" if emp=="Summacare"
replace emp="Sun Loan Company" if emp=="Sun Loan"
replace emp="Sun West Mortgage" if strpos(emp,"Sun West Mortgage")>0
replace emp="Sundance Resort" if strpos(emp,"Sundance Mountain Resort")==1
replace emp="Sungard Availability Services" if emp=="Sungard Availability Service"
replace emp="Sunglass Hut" if strpos(emp,"Sunglass Hut")==1
replace emp="Sutter Health" if strpos(emp,"Sutter Health")>0|emp=="Sutter Care At Home"|emp=="Sutter East Bay Medical Foundation"|emp=="Sutter Gould Medical Foundation"|emp=="Sutter Lakeside Hospital"|strpos(emp,"Sutter Medical")==1|emp=="Sutter Pacific Medical Foundation"|emp=="Sutter Physician Services"|emp=="Sutter Roseville Medical Center"|emp=="Sutter Tracy Community Hospital"|emp=="Sutter Shared Services"
replace emp="Svb Financial Group" if emp=="Svb "
replace emp="Swedish Covenant Hospital" if emp=="Swedish Covernant Hospital" | emp=="Swedish Covenent Hospital"
replace emp="Swift Transportation" if strpos(emp,"Swift Transportation")==1
replace emp="Swissport" if strpos(emp,"Swissport")==1
replace emp="T Rowe Price" if strpos(emp,"T Rowe Price")==1
replace emp="Take Care Health Systems" if emp=="Take Care Health System"
replace emp="Tallahassee Memorial Healthcare" if emp=="Tallahassee Memorial Hospital"
replace emp="Tanner Health System" if emp=="Tanner Medical Center"
replace emp="Tate Lyle" if emp=="Tate And Lyle"
replace emp="Technip" if emp=="Technip Usa"
replace emp="Telligen" if strpos(emp,"Telligen ")==1
replace emp="Tennessee Nashville Centennial Medical" if strpos(emp,"Tennessee Nashville Centennial Medical")==1
replace emp="Texas Webster Clear Lake Regional Medical Center" if strpos(emp,"Texas Webster Clear Lake Regional Med")==1
replace emp="The Fresh Market" if emp=="The Fresh Marke"
replace emp="Estee Lauder Company" if emp=="The Est Aoe Lauder Companies"
replace emp="Conam" if strpos(emp,"Conam Management")==1 | strpos(emp,"Conam Group Of Companies")>0
replace emp="The Herb Chambers Companies" if strpos(emp,"Herb Chambers")>0
replace emp="The Lego Group" if emp=="Lego Systems"
replace emp="The Retail Outsource" if strpos(emp,"The Retail Outsource")==1
replace emp="Thomas Jefferson University Hospital" if emp=="Thomas Jefferson University Hospitals"
replace emp="Thorsonbakerassociates" if emp=="Thorsonbakerassociatesinc"
replace emp="Tibco" if strpos(emp,"Tibco Software")==1
replace emp="Tmx Finance" if strpos(emp,"Tmx Finance")==1
replace emp="Transperfect Translations" if strpos(emp,"Transperfect Translations")==1
replace emp="Trimac Transportation" if emp=="Trimac"
replace emp="Tristar Health System" if emp=="Tristar Health"
replace emp="Triumph Aerostructures Vought Aircraft Division" if strpos(emp,"Triumph Aerostructures Vought")==1
replace emp="Trizetto" if emp=="Trizetto Group"
replace emp="Trump Hotels" if strpos(emp,"Trump")>0 & strpos(emp,"Hotel")>0
replace emp="Trump Hotels" if strpos(emp,"Trump")>0 & strpos(emp,"Resort")>0
replace emp="Trump Hotels" if emp=="Trump National Doral Miami"
replace emp="Tufts Medical Center" if emp=="Tufts New England Medical Center And Floating Hospit"
replace emp="Ty Cobb Healthcare" if strpos(emp,"Ty Cobb")==1 & strpos(emp,"Health")>0
replace emp="Ubisoft" if strpos(emp,"Ubisoft Entertainment")>0
replace emp="Ucla Health" if strpos(emp,"Ucla Health")==1|emp=="Ucla Medical Center"|emp=="Ucla Outpatient Clinics"
replace emp="Uniqlo" if strpos(emp,"Uniqlo ")==1
replace emp="United American Insurance Company" if strpos(emp,"United American Insurance")==1
replace emp="Univar" if emp=="Univar Usa "
replace emp="Universal Health Services Inc" if emp=="Universal Health Services 2C"
replace emp="Universal Orlando" if strpos(emp,"Universal Orlando")==1
replace emp="University Of Texas Health Science Center" if strpos(emp,"University Of Texas Health Science Center")==1
replace emp="Upper Chesapeake Health" if emp=="Upper Chesapeak Health"|emp=="Upper Chesapeake Medical Center"
replace emp="Utc Aerospace" if strpos(emp,"Utc Aerospace")==1
replace emp="Uw Hospitals Clinics" if emp=="Uw Hospital And Clinics"
replace emp="Vail Resort" if strpos(emp,"Vail")==1 & strpos(emp,"Resort")>0
replace emp="Valleycrest Companies" if strpos(emp,"Valleycrest Landscap")==1
replace emp="Van Wyk Company" if emp=="Van Wyk "
replace emp="Vcu Health System" if strpos(emp,"Vcu Health System")==1
replace emp="Veolia" if strpos(emp,"Veolia ")==1
replace emp="Verifone" if strpos(emp,"Verifone")==1
replace emp="Vidant Medical Group" if strpos(emp,"Vidant Medical")==1|emp=="Vidant Health"
replace emp="Virginia Mason Medical Center" if strpos(emp,"Virginia Mason Medical")==1
replace emp="Visionworks" if strpos(emp,"Visionworks ")==1
replace emp="Vistronix" if strpos(emp,"Vistronix ")==1
replace emp="Vitas Healthcare" if strpos(emp,"Vitas Innovative Hospice Care")==1
replace emp="Waggener Edstrom" if strpos(emp,"Waggener Edstrom")==1
replace emp="Waldorf Astoria" if strpos(emp,"Waldorf Astoria")==1
replace emp="Waste Industries" if emp=="Waste Industries Usa "
replace emp="Waste Pro" if emp=="Wastepro Usa"
replace emp="Wells Fargo" if strpos(emp,"Wells Fargo")==1|strpos(emp,"Wellsfargo")==1
replace emp="Wesbanco" if emp=="Wesbanco Bank "
replace emp="West Georgia Health System" if emp=="West Georgia Health Systesm"
replace emp="Westchester Medical Centers" if emp=="Westchester Medical Center"
replace emp="Western Connecticut Health Network" if strpos(emp,"Western Connecticut Health Network")==1|emp=="Western Connecticut Healthcare"
replace emp="Western Dental" if strpos(emp,"Western Dental ")==1
replace emp="Westgate Resorts" if strpos(emp,"Westgate")==1 & strpos(emp,"Resort")>0
replace emp="Westlake Financial" if strpos(emp,"Westlake Financial")==1
replace emp="Whelan Security" if strpos(emp,"Whelan Security")==1
replace emp="White Plains Hospital" if strpos(emp,"White Plains Hospital")==1
replace emp="Whitepages" if emp=="Whitepagesinc"
replace emp="William Blair And Company" if emp=="William Blair"
replace emp="Willis Group" if emp=="Willis Group Holdings"
replace emp="WillisKnighton Health System" if emp=="WillisKnighton Health Systems"
replace emp="Wyndham Worldwide" if strpos(emp,"Wyndham")==1
replace emp="Xanterra" if strpos(emp,"Xanterra")==1
replace emp="Xyratex" if emp=="Xyratex International"
replace emp="Yum Brands" if strpos(emp,"Yum Pizza Hut")==1
replace emp="Zions Bank" if emp=="Zions Bancorporation"
replace emp="Zippys" if strpos(emp,"Zippys Restaurant")==1
replace emp="Ziprealty" if strpos(emp,"Ziprealty Residential")==1
replace emp="Zynga" if strpos(emp,"Zynga")==1
replace emp=subinstr(emp,"Bancorp","Bank",.) if substr(emp, -8, 8)==" Bancorp"
replace emp="Aventura Hospital And Medical Center" if strpos(emp,"Aventura Hospital And Medical")==1
replace emp="Eisai" if strpos(emp,"Eisai ")==1
replace emp="International Cruise and Excursions" if strpos(emp,"International Cruise and Excursions")==1
replace emp="Ivy Rehab" if strpos(emp,"Ivy Rehab")==1
replace emp="Jack Kelley" if strpos(emp,"Jack Kelley")==1
replace emp="Its Just Lunch" if strpos(emp,"Its Just Lunch")==1
replace emp="Flint Hills Resources" if strpos(emp,"Flint Hills Resources")==1
replace emp="Foth" if strpos(emp,"Foth Production Solutions")==1
replace emp="Fred Meyer" if strpos(emp,"Fred Meyer ")==1
replace emp="Ftd Companies" if strpos(emp,"Ftd")==1
replace emp="Healthport" if strpos(emp,"Healthport Tech")==1
replace emp="Hines Interests" if strpos(emp,"Hines Interests")==1
replace emp="Immunotek Bio Centers" if strpos(emp,"Immunotek Bio Center")==1
replace emp="Insigma" if strpos(emp,"Insigmainc")==1
replace emp="International Millenium Consultants" if strpos(emp,"International Millenium Consultants")==1
replace emp="Keyence" if strpos(emp,"Keyence ")==1
replace emp="Kum Go" if strpos(emp,"Kum And Go")==1
replace emp="Kehe Distributors" if strpos(emp,"Kehe Distribution")==1
replace emp="JD Block And Associates" if strpos(emp,"JD Block Amp Associates")==1
replace emp="Jb Consulting" if strpos(emp,"Jb Consulting Systems")==1
replace emp="Jbn Associates" if strpos(emp,"Jbn And Associates")==1
replace emp="Lewis Resources Management" if strpos(emp,"Lewis Resources Management")==1
replace emp="Luk" if strpos(emp,"Luk Usa")==1
replace emp="Louis Dreyfus" if strpos(emp,"Louis Dreyfus")==1
replace emp="Mid America Apartment Commmunities" if strpos(emp,"Mid America Apartment")==1
replace emp="National Rural Utilities Cooperative Finance" if strpos(emp,"National Rural Utilities Cooperative Finance")==1
replace emp="Redbox" if strpos(emp,"Redbox Automated Retail")==1
replace emp="Quantum Integrators" if strpos(emp,"Quantum Integrators Group")==1
replace emp="Redfishtechnology" if strpos(emp,"Redfishtech")==1
replace emp="Robert Baird Company" if strpos(emp,"Robert Baird And Company")==1
replace emp="Sarnova" if strpos(emp,"Sarnova Hc")==1
replace emp="Shc Thompson" if strpos(emp,"Shc Thompson Medical Center")==1
replace emp="Sovereign Medical" if strpos(emp,"Sovereign Medical Group")==1

*Drop non-real emps
replace emp="NA" if strpos(emp,"Full Time")==1
replace emp="NA" if strpos(emp,"Full Training Provided")==1
replace emp="NA" if strpos(emp,"Hiring Asap")==1
replace emp="NA" if strpos(emp,"Hiring Immediately")==1
replace emp="NA" if strpos(emp,"Hiring Now")==1
replace emp="NA" if strpos(emp,"Now Hiring")==1
replace emp="NA" if strpos(emp,"not Provided")==1
replace emp="NA" if strpos(emp,"Immediate Availability")==1
replace emp="NA" if strpos(emp,"Immediate Entry Level Openings")==1
replace emp="NA" if strpos(emp,"Immediate Full Time Hire")==1
replace emp="NA" if strpos(emp,"Immediate Hire")==1
replace emp="NA" if strpos(emp,"Interviewing This Week")==1
replace emp="NA" if strpos(emp,"Interviewing Today")==1
replace emp="NA" if emp=="No Nights No Weekends"
replace emp="NA" if emp=="Non Disclosed"
replace emp="NA" if emp=="Not Mentioned"
replace emp="NA" if emp=="Paid Training Immediate Hire"
replace emp="NA" if strpos(emp,"Registrant Log In")>0
replace emp="NA" if strpos(emp,"Removed Due To Bad Name")>0
replace emp="NA" if strpos(emp,"Start Immediately")>0

* Firm name cleaning code from R
*Detect and correct employer names with some strange characters
*From Joe:
replace emp = "CRST International" if substr(emp, 1, strlen("Crst")) == "Crst"
replace emp = "Allegis" if substr(emp, 1, strlen("Allegis")) == "Allegis"
replace emp = "Tmc Transportation" if substr(emp, 1, strlen("Tmc")) == "Tmc"
replace emp = "K & B Transportation" if substr(emp, 1, strlen("Kb Transportation")) == "Kb Transportation"
replace emp = "Ew Wylie Corporation" if substr(emp, 1, strlen("Ew Wylie Flatbed")) == "Ew Wylie Flatbed"
replace emp = "Mercy Regional Medical Center" if substr(emp, 1, strlen("Mercy Regional Medical Center")) == "Mercy Regional Medical Center"
*From Isaac (made with regex, rather than substr, because these are written to search the whole string)
*These correct "Zz<emp name>" style errors
*There are actually a few legit employer names that start with Zz (Zzperformance auto parts for example), so this a casewise correction
replace emp = "Fred Finch Youth Center" if ustrregexm(emp, "Fred Finch Youth Center")
replace emp = "Dyouville College" if ustrregexm(emp, "Dyouville College")
replace emp = "State Ave Tire Auto Service" if ustrregexm(emp, "state Ave Tire Auto Service")

*Correct another group of firm names
local firm_names = `""Anderson Trucking" "Melton Truck Lines" "Western Express" "Forward Air" "Securitas""'
foreach firm_name in `firm_names' {
	replace emp = "`firm_name'" if substr(emp, 1, strlen("`firm_name'")) == "`firm_name'"
}

*Cut out leading apostrophes
replace emp = substr(emp, 2, .) if substr(emp, 1, 1) == "'"

*Set names to lowercase and remove internal multiple spaces (equivalent of str_squish)
replace emp=strlower(emp)
replace emp=stritrim(emp)

replace emp=subinstr(emp, " / this announcement covers navy and marines positions", "", .)
replace emp=subinstr(emp, "centers", "center", .)
replace emp=subinstr(emp, " incorporated", "", .)
replace emp=subinstr(emp, ", inc.", "", .)
replace emp=subinstr(emp, " inc.", "", .)
replace emp=subinstr(emp, ", inc", "", .)
replace emp=subinstr(emp, ", llc", "", .)
replace emp=subinstr(emp, " llc", "", .)
replace emp=subinstr(emp, " pllc", "", .)
replace emp=subinstr(emp, ", l p", "", .)
replace emp=subinstr(emp, " l p", "", .)
replace emp=subinstr(emp, "l l c", "", .)
replace emp=subinstr(emp, " lp", "", .)
replace emp=subinstr(emp, " limited", "", .)
replace emp=subinstr(emp, ", ltd", "", .)
replace emp=subinstr(emp, " ltd", "", .)
replace emp=subinstr(emp, ", l l c", "", .)
replace emp=subinstr(emp, " inc", "", .)
replace emp=subinstr(emp, ",inc", "", .)
replace emp=subinstr(emp, ",llc", "", .)
replace emp=subinstr(emp, ", llp", "", .)
replace emp=subinstr(emp, " llp", "", .)
replace emp=subinstr(emp, "plc", "", .)
replace emp=subinstr(emp, " corporation", "", .)
replace emp=subinstr(emp, "corporate", "", .)
replace emp=subinstr(emp, "the ", "", .)
replace emp=subinstr(emp, "company", "", .)
replace emp=subinstr(emp, " co ", "", .)
replace emp=ustrregexrf(emp, " co$", "", .)
replace emp=subinstr(emp, " corp", "", .)
replace emp=subinstr(emp, " stores", "", .)
replace emp=subinstr(emp, " store", "", .)
replace emp=subinstr(emp, "co llc", "", .)
replace emp=subinstr(emp, "&", "and", .)
replace emp=subinstr(emp, " of ", " ", .)
replace emp=subinstr(emp, " and ", " ", .)

*more stop words that i'm a little less confident in
replace emp=subinstr(emp, "distribution center", "", .)
replace emp=subinstr(emp, "services", "", .)

*from patent cleaning material
*want to go to upper case and pad ends of strings with spaces
replace emp = strupper(emp)
replace emp = " " + emp + " "
replace emp = subinstr( emp, " RES AND DEV ", " RANDD ", 1)
replace emp = subinstr( emp, " RECH AND DEV ", " RANDD ", 1)
replace emp = subinstr( emp," A B "," AB ",1)
replace emp = subinstr( emp," A CALIFORNIA CORP "," CORP ",1)
replace emp = subinstr( emp," A DELAWARE CORP "," CORP ",1)
replace emp = subinstr( emp," AKTIEBOLAGET "," AB ",1)
replace emp = subinstr( emp," AKTIEBOLAG "," AB ",1)
replace emp = subinstr( emp," ACADEMY "," ACAD ",1)
replace emp = subinstr( emp," ACTIEN GESELLSCHAFT "," AG ",1)
replace emp = subinstr( emp," ACTIENGESELLSCHAFT "," AG ",1)
replace emp = subinstr( emp," AKTIEN GESELLSCHAFT "," AG ",1)
replace emp = subinstr( emp," AKTIENGESELLSCHAFT "," AG ",1)
replace emp = subinstr( emp," AGRICOLAS "," AGRIC ",1)
replace emp = subinstr( emp," AGRICOLA "," AGRIC ",1)
replace emp = subinstr( emp," AGRICOLES "," AGRIC ",1)
replace emp = subinstr( emp," AGRICOLE "," AGRIC ",1)
replace emp = subinstr( emp," AGRICOLI "," AGRIC ",1)
replace emp = subinstr( emp," AGRICOLTURE "," AGRIC ",1)
replace emp = subinstr( emp," AGRICULTURA "," AGRIC ",1)
replace emp = subinstr( emp," AGRICULTURAL "," AGRIC ",1)
replace emp = subinstr( emp," AGRICULTURE "," AGRIC ",1)
replace emp = subinstr( emp," AKADEMIA "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIEI "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIE "," AKAD ",1)
replace emp = subinstr( emp," AKADEMII "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIJA "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIYA "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIYAKH "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIYAM "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIYAMI "," AKAD ",1)
replace emp = subinstr( emp," AKADEMIYU "," AKAD ",1)
replace emp = subinstr( emp," AKADEMI "," AKAD ",1)
replace emp = subinstr( emp," ALLGEMEINER "," ALLG ",1)
replace emp = subinstr( emp," ALLGEMEINE "," ALLG ",1)
replace emp = subinstr( emp," ANTREPRIZA "," ANTR ",1)
replace emp = subinstr( emp," APARARII "," APAR ",1)
replace emp = subinstr( emp," APARATELOR "," APAR ",1)
replace emp = subinstr( emp," APPARATEBAU "," APP ",1)
replace emp = subinstr( emp," APPARATUS "," APP ",1)
replace emp = subinstr( emp," APPARECHHI "," APP ",1)
replace emp = subinstr( emp," APPAREILLAGES "," APP ",1)
replace emp = subinstr( emp," APPAREILLAGE "," APP ",1)
replace emp = subinstr( emp," APPAREILS "," APP ",1)
replace emp = subinstr( emp," APPAREIL "," APP ",1)
replace emp = subinstr( emp," APARATE "," APAR ",1)
replace emp = subinstr( emp," APPARATE "," APP ",1)
replace emp = subinstr( emp," APPLICATIONS "," APPL ",1)
replace emp = subinstr( emp," APPLICATION "," APPL ",1)
replace emp = subinstr( emp," APPLICAZIONE "," APPL ",1)
replace emp = subinstr( emp," APPLICAZIONI "," APPL ",1)
replace emp = subinstr( emp," ANPARTSSELSKABET "," APS ",1)
replace emp = subinstr( emp," ANPARTSSELSKAB "," APS ",1)
replace emp = subinstr( emp," A/S "," AS ",1)
replace emp = subinstr( emp," AKTIESELSKABET "," AS ",1)
replace emp = subinstr( emp," AKTIESELSKAB "," AS ",1)
replace emp = subinstr( emp," ASSOCIACAO "," ASSOC ",1)
replace emp = subinstr( emp," ASSOCIATED "," ASSOC ",1)
replace emp = subinstr( emp," ASSOCIATES "," ASSOCIATES ",1)
replace emp = subinstr( emp," ASSOCIATE "," ASSOCIATES ",1)
replace emp = subinstr( emp," ASSOCIATION "," ASSOC ",1)
replace emp = subinstr( emp," BETEILIGUNGSGESELLSCHAFT MBH "," BET GMBH ",1)
replace emp = subinstr( emp," BETEILIGUNGS GESELLSCHAFT MIT "," BET GMBH ",1)
replace emp = subinstr( emp," BETEILIGUNGSGESELLSCHAFT "," BET GES ",1)
replace emp = subinstr( emp," BESCHRANKTER HAFTUNG "," BET GMBH ",1)
replace emp = subinstr( emp," BROEDERNA "," BRDR ",1)
replace emp = subinstr( emp," BROEDRENE "," BRDR ",1)
replace emp = subinstr( emp," BRODERNA "," BRDR ",1)
replace emp = subinstr( emp," BRODRENE "," BRDR ",1)
replace emp = subinstr( emp," BROTHERS "," BROS ",1)
replace emp = subinstr( emp," BESLOTEN VENNOOTSCHAP MET "," BV ",1)
replace emp = subinstr( emp," BESLOTEN VENNOOTSCHAP "," BV ",1)
replace emp = subinstr( emp," BEPERKTE AANSPRAKELIJKHEID "," BV ",1)
replace emp = subinstr( emp," CLOSE CORPORATION "," CC ",1)
replace emp = subinstr( emp," CENTER "," CENT ",1)
replace emp = subinstr( emp," CENTRAAL "," CENT ",1)
replace emp = subinstr( emp," CENTRALA "," CENT ",1)
replace emp = subinstr( emp," CENTRALES "," CENT ",1)
replace emp = subinstr( emp," CENTRALE "," CENT ",1)
replace emp = subinstr( emp," CENTRAL "," CENT ",1)
replace emp = subinstr( emp," CENTRAUX "," CENT ",1)
replace emp = subinstr( emp," CENTRE "," CENT ",1)
replace emp = subinstr( emp," CENTRO "," CENT ",1)
replace emp = subinstr( emp," CENTRUL "," CENT ",1)
replace emp = subinstr( emp," CENTRUM "," CENT ",1)
replace emp = subinstr( emp," CERCETARE "," CERC ",1)
replace emp = subinstr( emp," CERCETARI "," CERC ",1)
replace emp = subinstr( emp," CHEMICALS "," CHEM ",1)
replace emp = subinstr( emp," CHEMICAL "," CHEM ",1)
replace emp = subinstr( emp," CHEMICKEJ "," CHEM ",1)
replace emp = subinstr( emp," CHEMICKE "," CHEM ",1)
replace emp = subinstr( emp," CHEMICKYCH "," CHEM ",1)
replace emp = subinstr( emp," CHEMICKY "," CHEM ",1)
replace emp = subinstr( emp," CHEMICZNE "," CHEM ",1)
replace emp = subinstr( emp," CHEMICZNY "," CHEM ",1)
replace emp = subinstr( emp," CHEMIE "," CHEM ",1)
replace emp = subinstr( emp," CHEMII "," CHEM ",1)
replace emp = subinstr( emp," CHEMISCHE "," CHEM ",1)
replace emp = subinstr( emp," CHEMISCH "," CHEM ",1)
replace emp = subinstr( emp," CHEMISKEJ "," CHEM ",1)
replace emp = subinstr( emp," CHEMISTRY "," CHEM ",1)
replace emp = subinstr( emp," CHIMICA "," CHIM ",1)
replace emp = subinstr( emp," CHIMICE "," CHIM ",1)
replace emp = subinstr( emp," CHIMICI "," CHIM ",1)
replace emp = subinstr( emp," CHIMICO "," CHIM ",1)
replace emp = subinstr( emp," CHIMIC "," CHIM ",1)
replace emp = subinstr( emp," CHIMIEI "," CHIM ",1)
replace emp = subinstr( emp," CHIMIE "," CHIM ",1)
replace emp = subinstr( emp," CHIMIESKOJ "," CHIM ",1)
replace emp = subinstr( emp," CHIMII "," CHIM ",1)
replace emp = subinstr( emp," CHIMIKO "," CHIM ",1)
replace emp = subinstr( emp," CHIMIQUES "," CHIM ",1)
replace emp = subinstr( emp," CHIMIQUE "," CHIM ",1)
replace emp = subinstr( emp," CHIMIYAKH "," CHIM ",1)
replace emp = subinstr( emp," CHIMIYAMI "," CHIM ",1)
replace emp = subinstr( emp," CHIMIYAM "," CHIM ",1)
replace emp = subinstr( emp," CHIMIYA "," CHIM ",1)
replace emp = subinstr( emp," CHIMIYU "," CHIM ",1)
replace emp = subinstr( emp," COMPAGNIE FRANCAISE "," CIE FR ",1)
replace emp = subinstr( emp," COMPAGNIE GENERALE "," CIE GEN ",1)
replace emp = subinstr( emp," COMPAGNIE INDUSTRIALE "," CIE IND ",1)
replace emp = subinstr( emp," COMPAGNIE INDUSTRIELLE "," CIE IND ",1)
replace emp = subinstr( emp," COMPAGNIE INDUSTRIELLES "," CIE IND ",1)
replace emp = subinstr( emp," COMPAGNIE INTERNATIONALE "," CIE INT ",1)
replace emp = subinstr( emp," COMPAGNIE NATIONALE "," CIE NAT ",1)
replace emp = subinstr( emp," COMPAGNIE PARISIENNE "," CIE PARIS ",1)
replace emp = subinstr( emp," COMPAGNIE PARISIENN "," CIE PARIS ",1)
replace emp = subinstr( emp," COMPAGNIE PARISIEN "," CIE PARIS ",1)
replace emp = subinstr( emp," COMPANIES "," CO ",1)
replace emp = subinstr( emp," COMPAGNIA "," CIA ",1)
replace emp = subinstr( emp," COMPANHIA "," CIA ",1)
replace emp = subinstr( emp," COMPAGNIE "," CIE ",1)
replace emp = subinstr( emp," COMPANY "," CO ",1)
replace emp = subinstr( emp," COMBINATUL "," COMB ",1)
replace emp = subinstr( emp," COMMERCIALE "," COMML ",1)
replace emp = subinstr( emp," COMMERCIAL "," COMML ",1)
replace emp = subinstr( emp," CONSOLIDATED "," CONSOL ",1)
replace emp = subinstr( emp," CONSTRUCCIONES "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCCIONE "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCCION "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTIE "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTII "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTIILOR "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTIONS "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTION "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTORTUL "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTORUL "," CONSTR ",1)
replace emp = subinstr( emp," CONSTRUCTOR "," CONSTR ",1)
replace emp = subinstr( emp," CO OPERATIVES "," COOP ",1)
replace emp = subinstr( emp," CO OPERATIVE "," COOP ",1)
replace emp = subinstr( emp," COOPERATIEVE "," COOP ",1)
replace emp = subinstr( emp," COOPERATIVA "," COOP ",1)
replace emp = subinstr( emp," COOPERATIVES "," COOP ",1)
replace emp = subinstr( emp," COOPERATIVE "," COOP ",1)
replace emp = subinstr( emp," INCORPORATED "," INC ",1)
replace emp = subinstr( emp," INCORPORATION "," INC ",1)
replace emp = subinstr( emp," CORPORATE "," CORP ",1)
replace emp = subinstr( emp," CORPORATION OF AMERICA "," CORP ",1)
replace emp = subinstr( emp," CORPORATION "," CORP ",1)
replace emp = subinstr( emp," CORPORASTION "," CORP ",1)
replace emp = subinstr( emp," CORPORATIOON "," CORP ",1)
replace emp = subinstr( emp," COSTRUZIONI "," COSTR ",1)
replace emp = subinstr( emp," DEUTSCHEN "," DDR ",1)
replace emp = subinstr( emp," DEUTSCHE "," DDR ",1)
replace emp = subinstr( emp," DEMOKRATISCHEN REPUBLIK "," DDR ",1)
replace emp = subinstr( emp," DEMOKRATISCHE REPUBLIK "," DDR ",1)
replace emp = subinstr( emp," DEPARTEMENT "," DEPT ",1)
replace emp = subinstr( emp," DEPARTMENT "," DEPT ",1)
replace emp = subinstr( emp," DEUTSCHES "," DEUT ",1)
replace emp = subinstr( emp," DEUTSCHEN "," DEUT ",1)
replace emp = subinstr( emp," DEUTSCHER "," DEUT ",1)
replace emp = subinstr( emp," DEUTSCHLAND "," DEUT ",1)
replace emp = subinstr( emp," DEUTSCHE "," DEUT ",1)
replace emp = subinstr( emp," DEUTSCH "," DEUT ",1)
replace emp = subinstr( emp," DEVELOPMENTS "," DEV ",1)
replace emp = subinstr( emp," DEVELOPMENT "," DEV ",1)
replace emp = subinstr( emp," DEVELOPPEMENTS "," DEV ",1)
replace emp = subinstr( emp," DEVELOPPEMENT "," DEV ",1)
replace emp = subinstr( emp," DEVELOP "," DEV ",1)
replace emp = subinstr( emp," DIVISIONE "," DIV ",1)
replace emp = subinstr( emp," DIVISION "," DIV ",1)
replace emp = subinstr( emp," ENGINEERING "," ENG ",1)
replace emp = subinstr( emp," EQUIPEMENTS "," EQUIP ",1)
replace emp = subinstr( emp," EQUIPEMENT "," EQUIP ",1)
replace emp = subinstr( emp," EQUIPMENTS "," EQUIP ",1)
replace emp = subinstr( emp," EQUIPMENT "," EQUIP ",1)
replace emp = subinstr( emp," ESTABLISHMENTS "," ESTAB ",1)
replace emp = subinstr( emp," ESTABLISHMENT "," ESTAB ",1)
replace emp = subinstr( emp," ESTABLISSEMENTS "," ESTAB ",1)
replace emp = subinstr( emp," ESTABLISSEMENT "," ESTAB ",1)
replace emp = subinstr( emp," ETABLISSEMENTS "," ETAB ",1)
replace emp = subinstr( emp," ETABLISSEMENT "," ETAB ",1)
replace emp = subinstr( emp," ETABS "," ETAB ",1)
replace emp = subinstr( emp," ETS "," ETAB ",1)
replace emp = subinstr( emp," ETUDES "," ETUD ",1)
replace emp = subinstr( emp," ETUDE "," ETUD ",1)
replace emp = subinstr( emp," EUROPAEISCHEN "," EURO ",1)
replace emp = subinstr( emp," EUROPAEISCHES "," EURO ",1)
replace emp = subinstr( emp," EUROPAEISCHE "," EURO ",1)
replace emp = subinstr( emp," EUROPAISCHEN "," EURO ",1)
replace emp = subinstr( emp," EUROPAISCHES "," EURO ",1)
replace emp = subinstr( emp," EUROPAISCHE "," EURO ",1)
replace emp = subinstr( emp," EUROPEAN "," EURO ",1)
replace emp = subinstr( emp," EUROPEENNE "," EURO ",1)
replace emp = subinstr( emp," EUROPEEN "," EURO ",1)
replace emp = subinstr( emp," EUROPEA "," EURO ",1)
replace emp = subinstr( emp," EUROPE "," EURO ",1)
replace emp = subinstr( emp," EINGETRAGENER VEREIN "," EV ",1)
replace emp = subinstr( emp," EXPLOATERINGS "," EXPL ",1)
replace emp = subinstr( emp," EXPLOATERING "," EXPL ",1)
replace emp = subinstr( emp," EXPLOITATIE "," EXPL ",1)
replace emp = subinstr( emp," EXPLOITATIONS "," EXPL ",1)
replace emp = subinstr( emp," EXPLOITATION "," EXPL ",1)
replace emp = subinstr( emp," FIRMA "," FA ",1)
replace emp = subinstr( emp," FABBRICAZIONI "," FAB ",1)
replace emp = subinstr( emp," FABBRICHE "," FAB ",1)
replace emp = subinstr( emp," FABRICATIONS "," FAB ",1)
replace emp = subinstr( emp," FABRICATION "," FAB ",1)
replace emp = subinstr( emp," FABBRICA "," FAB ",1)
replace emp = subinstr( emp," FABRICA "," FAB ",1)
replace emp = subinstr( emp," FABRIEKEN "," FAB ",1)
replace emp = subinstr( emp," FABRIEK "," FAB ",1)
replace emp = subinstr( emp," FABRIKER "," FAB ",1)
replace emp = subinstr( emp," FABRIK "," FAB ",1)
replace emp = subinstr( emp," FABRIQUES "," FAB ",1)
replace emp = subinstr( emp," FABRIQUE "," FAB ",1)
replace emp = subinstr( emp," FABRIZIO "," FAB ",1)
replace emp = subinstr( emp," FABRYKA "," FAB ",1)
replace emp = subinstr( emp," FARMACEUTICA "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTICE "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTICHE "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTICI "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTICOS "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTICO "," FARM ",1)
replace emp = subinstr( emp," FARMACEUTISK "," FARM ",1)
replace emp = subinstr( emp," FARMACEVTSKIH "," FARM ",1)
replace emp = subinstr( emp," FARMACIE "," FARM ",1)
replace emp = subinstr( emp," FONDATION "," FOND ",1)
replace emp = subinstr( emp," FONDAZIONE "," FOND ",1)
replace emp = subinstr( emp," FOUNDATIONS "," FOUND ",1)
replace emp = subinstr( emp," FOUNDATION "," FOUND ",1)
replace emp = subinstr( emp," FRANCAISE "," FR ",1)
replace emp = subinstr( emp," FRANCAIS "," FR ",1)
replace emp = subinstr( emp," F LLI "," FRAT ",1)
replace emp = subinstr( emp," FLLI "," FRAT ",1)
replace emp = subinstr( emp," FRATELLI "," FRAT ",1)
replace emp = subinstr( emp," GEBRODERS "," GEBR ",1)
replace emp = subinstr( emp," GEBRODER "," GEBR ",1)
replace emp = subinstr( emp," GEBROEDERS "," GEBR ",1)
replace emp = subinstr( emp," GEBROEDER "," GEBR ",1)
replace emp = subinstr( emp," GEBRUDERS "," GEBR ",1)
replace emp = subinstr( emp," GEBRUDER "," GEBR ",1)
replace emp = subinstr( emp," GEBRUEDERS "," GEBR ",1)
replace emp = subinstr( emp," GEBRUEDER "," GEBR ",1)
replace emp = subinstr( emp," GEB "," GEBR ",1)
replace emp = subinstr( emp," GENERALA "," GEN ",1)
replace emp = subinstr( emp," GENERALES "," GEN ",1)
replace emp = subinstr( emp," GENERALE "," GEN ",1)
replace emp = subinstr( emp," GENERAL "," GEN ",1)
replace emp = subinstr( emp," GENERAUX "," GEN ",1)
replace emp = subinstr( emp," GESELLSCHAFT "," GES ",1)
replace emp = subinstr( emp," GEWERKSCHAFT "," GEW ",1)
replace emp = subinstr( emp," GAKKO HOJIN "," GH ",1)
replace emp = subinstr( emp," GAKKO HOUJIN "," GH ",1)
replace emp = subinstr( emp," GUTEHOFFNUNGSCHUETTE "," GHH ",1)
replace emp = subinstr( emp," GUTEHOFFNUNGSCHUTTE "," GHH ",1)
replace emp = subinstr( emp," GOMEI GAISHA "," GK ",1)
replace emp = subinstr( emp," GOMEI KAISHA "," GK ",1)
replace emp = subinstr( emp," GOSHI KAISHA "," GK ",1)
replace emp = subinstr( emp," GOUSHI GAISHA "," GK ",1)
replace emp = subinstr( emp," GESELLSCHAFT MBH "," GMBH ",1)
replace emp = subinstr( emp," GESELLSCHAFT MIT BESCHRANKTER HAFTUNG "," GMBH ",1)
replace emp = subinstr( emp," GROUPEMENT "," GRP ",1)
replace emp = subinstr( emp," GROUPMENT "," GRP ",1)
replace emp = subinstr( emp," HANDELSMAATSCHAPPIJ "," HANDL ",1)
replace emp = subinstr( emp," HANDELSMIJ "," HANDL ",1)
replace emp = subinstr( emp," HANDELS BOLAGET "," HB ",1)
replace emp = subinstr( emp," HANDELSBOLAGET "," HB ",1)
replace emp = subinstr( emp," HER MAJESTY THE QUEEN IN RIGHT OF CANADA AS REPRESENTED BY THE MINISTER OF "," CANADA MIN OF ",1)
replace emp = subinstr( emp," HER MAJESTY THE QUEEN "," UK ",1)
replace emp = subinstr( emp," INDUSTRIAS "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALS "," IND ",1)
replace emp = subinstr( emp," INDUSTRIAL "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALA "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALE "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALIZARE "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALIZAREA "," IND ",1)
replace emp = subinstr( emp," INDUSTRIALI "," IND ",1)
replace emp = subinstr( emp," INDUSTRIEELE "," IND ",1)
replace emp = subinstr( emp," INDUSTRIEI "," IND ",1)
replace emp = subinstr( emp," INDUSTRIELS "," IND ",1)
replace emp = subinstr( emp," INDUSTRIELLES "," IND ",1)
replace emp = subinstr( emp," INDUSTRIELLE "," IND ",1)
replace emp = subinstr( emp," INDUSTRIELL "," IND ",1)
replace emp = subinstr( emp," INDUSTRIEL "," IND ",1)
replace emp = subinstr( emp," INDUSTRIER "," IND ",1)
replace emp = subinstr( emp," INDUSTRIES "," IND ",1)
replace emp = subinstr( emp," INDUSTRII "," IND ",1)
replace emp = subinstr( emp," INDUSTRIJ "," IND ",1)
replace emp = subinstr( emp," INDUSTRIYAKH "," IND ",1)
replace emp = subinstr( emp," INDUSTRIYAM "," IND ",1)
replace emp = subinstr( emp," INDUSTRIYAMI "," IND ",1)
replace emp = subinstr( emp," INDUSTRIYA "," IND ",1)
replace emp = subinstr( emp," INDUSTRIYU "," IND ",1)
replace emp = subinstr( emp," INDUSTRIA "," IND ",1)
replace emp = subinstr( emp," INDUSTRIE "," IND ",1)
replace emp = subinstr( emp," INDUSTRI "," IND ",1)
replace emp = subinstr( emp," INDUSTRY "," IND ",1)
replace emp = subinstr( emp," INGENIERIA "," ING ",1)
replace emp = subinstr( emp," INGENIER "," ING ",1)
replace emp = subinstr( emp," INGENIEURS "," ING ",1)
replace emp = subinstr( emp," INGENIEURBUERO "," ING ",1)
replace emp = subinstr( emp," INGENIEURBUREAU "," ING ",1)
replace emp = subinstr( emp," INGENIEURBURO "," ING ",1)
replace emp = subinstr( emp," INGENIEURGESELLSCHAFT "," ING ",1)
replace emp = subinstr( emp," INGENIEURSBUREAU "," ING ",1)
replace emp = subinstr( emp," INGENIEURTECHNISCHES "," ING ",1)
replace emp = subinstr( emp," INGENIEURTECHNISCHE "," ING ",1)
replace emp = subinstr( emp," INGENIEUR "," ING ",1)
replace emp = subinstr( emp," INGENIOERFIRMAET "," ING ",1)
replace emp = subinstr( emp," INGENIORSFIRMAN "," ING ",1)
replace emp = subinstr( emp," INGENIORSFIRMA "," ING ",1)
replace emp = subinstr( emp," INGENJORSFIRMA "," ING ",1)
replace emp = subinstr( emp," INGINERIE "," ING ",1)
replace emp = subinstr( emp," INSTITUTE FRANCAISE "," INST FR ",1)
replace emp = subinstr( emp," INSTITUT FRANCAIS "," INST FR ",1)
replace emp = subinstr( emp," INSTITUTE NATIONALE "," INST NAT ",1)
replace emp = subinstr( emp," INSTITUT NATIONAL "," INST NAT ",1)
replace emp = subinstr( emp," INSTITUTAMI "," INST ",1)
replace emp = subinstr( emp," INSTITUTAMKH "," INST ",1)
replace emp = subinstr( emp," INSTITUTAM "," INST ",1)
replace emp = subinstr( emp," INSTITUTA "," INST ",1)
replace emp = subinstr( emp," INSTITUTES "," INST ",1)
replace emp = subinstr( emp," INSTITUTET "," INST ",1)
replace emp = subinstr( emp," INSTITUTE "," INST ",1)
replace emp = subinstr( emp," INSTITUTOM "," INST ",1)
replace emp = subinstr( emp," INSTITUTOV "," INST ",1)
replace emp = subinstr( emp," INSTITUTO "," INST ",1)
replace emp = subinstr( emp," INSTITUTT "," INST ",1)
replace emp = subinstr( emp," INSTITUTUL "," INST ",1)
replace emp = subinstr( emp," INSTITUTU "," INST ",1)
replace emp = subinstr( emp," INSTITUTY "," INST ",1)
replace emp = subinstr( emp," INSTITUT "," INST ",1)
replace emp = subinstr( emp," INSTITUUT "," INST ",1)
replace emp = subinstr( emp," INSTITZHT "," INST ",1)
replace emp = subinstr( emp," INSTYTUT "," INST ",1)
replace emp = subinstr( emp," INSINOORITOMISTO "," INSTMSTO ",1)
replace emp = subinstr( emp," INSTRUMENTS "," INSTR ",1)
replace emp = subinstr( emp," INSTRUMENTATION "," INSTR ",1)
replace emp = subinstr( emp," INSTRUMENTE "," INSTR ",1)
replace emp = subinstr( emp," INSTRUMENT "," INSTR ",1)
replace emp = subinstr( emp," INTERNATL "," INT ",1)
replace emp = subinstr( emp," INTERNACIONAL "," INT ",1)
replace emp = subinstr( emp," INTERNATIONAL "," INT ",1)
replace emp = subinstr( emp," INTERNATIONALEN "," INT ",1)
replace emp = subinstr( emp," INTERNATIONALE "," INT ",1)
replace emp = subinstr( emp," INTERNATIONAUX "," INT ",1)
replace emp = subinstr( emp," INTERNATIONELLA "," INT ",1)
replace emp = subinstr( emp," INTERNAZIONALE "," INT ",1)
replace emp = subinstr( emp," INTL "," INT ",1)
replace emp = subinstr( emp," INTREPRINDEREA "," INTR ",1)
replace emp = subinstr( emp," ISTITUTO "," IST ",1)
replace emp = subinstr( emp," ITALIANA "," ITAL ",1)
replace emp = subinstr( emp," ITALIANE "," ITAL ",1)
replace emp = subinstr( emp," ITALIANI "," ITAL ",1)
replace emp = subinstr( emp," ITALIANO "," ITAL ",1)
replace emp = subinstr( emp," ITALIENNE "," ITAL ",1)
replace emp = subinstr( emp," ITALIEN "," ITAL ",1)
replace emp = subinstr( emp," ITALIAN "," ITAL ",1)
replace emp = subinstr( emp," ITALIA "," ITAL ",1)
replace emp = subinstr( emp," ITALI "," ITAL ",1)
replace emp = subinstr( emp," ITALO "," ITAL ",1)
replace emp = subinstr( emp," ITALY "," ITAL ",1)
replace emp = subinstr( emp," JUNIOR "," JR ",1)
replace emp = subinstr( emp," KOMMANDIT BOLAG "," KB ",1)
replace emp = subinstr( emp," KOMMANDIT BOLAGET "," KB ",1)
replace emp = subinstr( emp," KOMMANDITBOLAGET "," KB ",1)
replace emp = subinstr( emp," KOMMANDITBOLAG "," KB ",1)
replace emp = subinstr( emp," KOMMANDIT GESELLSCHAFT "," KG ",1)
replace emp = subinstr( emp," KOMMANDITGESELLSCHAFT "," KG ",1)
replace emp = subinstr( emp," KOMMANDIT GESELLSCHAFT AUF AKTIEN "," KGAA ",1)
replace emp = subinstr( emp," KOMMANDITGESELLSCHAFT AUF AKTIEN "," KGAA ",1)
replace emp = subinstr( emp," KUTATO INTEZETE "," KI ",1)
replace emp = subinstr( emp," KUTATO INTEZET "," KI ",1)
replace emp = subinstr( emp," KUTATOINTEZETE "," KI ",1)
replace emp = subinstr( emp," KUTATOINTEZET "," KI ",1)
replace emp = subinstr( emp," KABUSHIKI GAISHA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKI KAISHA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKI GAISYA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKI KAISYA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKIGAISHA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKIKAISHA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKIGAISYA "," KK ",1)
replace emp = subinstr( emp," KABUSHIKIKAISYA "," KK ",1)
replace emp = subinstr( emp," KOMBINATU "," KOMB ",1)
replace emp = subinstr( emp," KOMBINATY "," KOMB ",1)
replace emp = subinstr( emp," KOMBINAT "," KOMB ",1)
replace emp = subinstr( emp," KONINKLIJKE "," KONINK ",1)
replace emp = subinstr( emp," KONCERNOVY PODNIK "," KP ",1)
replace emp = subinstr( emp," KUNSTSTOFFTECHNIK "," KUNST ",1)
replace emp = subinstr( emp," KUNSTSTOFF "," KUNST ",1)
replace emp = subinstr( emp," LABORATOIRES "," LAB ",1)
replace emp = subinstr( emp," LABORATOIRE "," LAB ",1)
replace emp = subinstr( emp," LABORATOIR "," LAB ",1)
replace emp = subinstr( emp," LABORATORIEI "," LAB ",1)
replace emp = subinstr( emp," LABORATORIES "," LAB ",1)
replace emp = subinstr( emp," LABORATORII "," LAB ",1)
replace emp = subinstr( emp," LABORATORIJ "," LAB ",1)
replace emp = subinstr( emp," LABORATORIOS "," LAB ",1)
replace emp = subinstr( emp," LABORATORIO "," LAB ",1)
replace emp = subinstr( emp," LABORATORIUM "," LAB ",1)
replace emp = subinstr( emp," LABORATORI "," LAB ",1)
replace emp = subinstr( emp," LABORATORY "," LAB ",1)
replace emp = subinstr( emp," LABORTORI "," LAB ",1)
replace emp = subinstr( emp," LAVORAZA "," LAVORAZ ",1)
replace emp = subinstr( emp," LAVORAZIONE "," LAVORAZ ",1)
replace emp = subinstr( emp," LAVORAZIONI "," LAVORAZ ",1)
replace emp = subinstr( emp," LAVORAZIO "," LAVORAZ ",1)
replace emp = subinstr( emp," LAVORAZI "," LAVORAZ ",1)
replace emp = subinstr( emp," LIMITED PARTNERSHIP "," LP ",1)
replace emp = subinstr( emp," LIMITED "," LTD ",1)
replace emp = subinstr( emp," LTD LTEE "," LTD ",1)
replace emp = subinstr( emp," MASCHINENVERTRIEB "," MASCH ",1)
replace emp = subinstr( emp," MASCHINENBAUANSTALT "," MASCHBAU ",1)
replace emp = subinstr( emp," MASCHINENBAU "," MASCHBAU ",1)
replace emp = subinstr( emp," MASCHINENFABRIEK "," MASCHFAB ",1)
replace emp = subinstr( emp," MASCHINENFABRIKEN "," MASCHFAB ",1)
replace emp = subinstr( emp," MASCHINENFABRIK "," MASCHFAB ",1)
replace emp = subinstr( emp," MASCHINENFAB "," MASCHFAB ",1)
replace emp = subinstr( emp," MASCHINEN "," MASCH ",1)
replace emp = subinstr( emp," MASCHIN "," MASCH ",1)
replace emp = subinstr( emp," MIT BESCHRANKTER HAFTUNG "," MBH ",1)
replace emp = subinstr( emp," MANUFACTURINGS "," MFG ",1)
replace emp = subinstr( emp," MANUFACTURING "," MFG ",1)
replace emp = subinstr( emp," MANIFATTURAS "," MFR ",1)
replace emp = subinstr( emp," MANIFATTURA "," MFR ",1)
replace emp = subinstr( emp," MANIFATTURE "," MFR ",1)
replace emp = subinstr( emp," MANUFACTURAS "," MFR ",1)
replace emp = subinstr( emp," MANUFACTURERS "," MFR ",1)
replace emp = subinstr( emp," MANUFACTURER "," MFR ",1)
replace emp = subinstr( emp," MANUFACTURES "," MFR ",1)
replace emp = subinstr( emp," MANUFACTURE "," MFR ",1)
replace emp = subinstr( emp," MANUFATURA "," MFR ",1)
replace emp = subinstr( emp," MAATSCHAPPIJ "," MIJ ",1)
replace emp = subinstr( emp," MEDICAL "," MED ",1)
replace emp = subinstr( emp," MINISTERE "," MIN ",1)
replace emp = subinstr( emp," MINISTERIUM "," MIN ",1)
replace emp = subinstr( emp," MINISTERO "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVAKH "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVAM "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVAMI "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVA "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVE "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVO "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVOM "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTVU "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTV "," MIN ",1)
replace emp = subinstr( emp," MINISTERSTWO "," MIN ",1)
replace emp = subinstr( emp," MINISTERUL "," MIN ",1)
replace emp = subinstr( emp," MINISTRE "," MIN ",1)
replace emp = subinstr( emp," MINISTRY "," MIN ",1)
replace emp = subinstr( emp," MINISTER "," MIN ",1)
replace emp = subinstr( emp," MAGYAR TUDOMANYOS AKADEMIA "," MTA ",1)
replace emp = subinstr( emp," NATIONAAL "," NAT ",1)
replace emp = subinstr( emp," NATIONAL "," NAT ",1)
replace emp = subinstr( emp," NATIONALE "," NAT ",1)
replace emp = subinstr( emp," NATIONAUX "," NAT ",1)
replace emp = subinstr( emp," NATL "," NAT ",1)
replace emp = subinstr( emp," NAZIONALE "," NAZ ",1)
replace emp = subinstr( emp," NAZIONALI "," NAZ ",1)
replace emp = subinstr( emp," NORDDEUTSCH "," NORDDEUT ",1)
replace emp = subinstr( emp," NORDDEUTSCHE "," NORDDEUT ",1)
replace emp = subinstr( emp," NORDDEUTSCHER "," NORDDEUT ",1)
replace emp = subinstr( emp," NORDDEUTSCHES "," NORDDEUT ",1)
replace emp = subinstr( emp," NARODNI PODNIK "," NP ",1)
replace emp = subinstr( emp," NARODNIJ PODNIK "," NP ",1)
replace emp = subinstr( emp," NARODNY PODNIK "," NP ",1)
replace emp = subinstr( emp," NAAMLOOSE VENOOTSCHAP "," NV ",1)
replace emp = subinstr( emp," NAAMLOZE VENNOOTSCHAP "," NV ",1)
replace emp = subinstr( emp," N V "," NV ",1)
replace emp = subinstr( emp," OESTERREICHISCHES "," OESTERR ",1)
replace emp = subinstr( emp," OESTERREICHISCHE "," OESTERR ",1)
replace emp = subinstr( emp," OESTERREICHISCH "," OESTERR ",1)
replace emp = subinstr( emp," OESTERREICH "," OESTERR ",1)
replace emp = subinstr( emp," OSTERREICHISCHES "," OESTERR ",1)
replace emp = subinstr( emp," OSTERREICHISCHE "," OESTERR ",1)
replace emp = subinstr( emp," OSTERREICHISCH "," OESTERR ",1)
replace emp = subinstr( emp," OSTERREICH "," OESTERR ",1)
replace emp = subinstr( emp," OFFICINE MECCANICA "," OFF MEC ",1)
replace emp = subinstr( emp," OFFICINE MECCANICHE "," OFF MEC ",1)
replace emp = subinstr( emp," OFFICINE NATIONALE "," OFF NAT ",1)
replace emp = subinstr( emp," OFFENE HANDELSGESELLSCHAFT "," OHG ",1)
replace emp = subinstr( emp," ONTWIKKELINGSBUREAU "," ONTWIK ",1)
replace emp = subinstr( emp," ONTWIKKELINGS "," ONTWIK ",1)
replace emp = subinstr( emp," OBOROVY PODNIK "," OP ",1)
replace emp = subinstr( emp," ORGANISATIE "," ORG ",1)
replace emp = subinstr( emp," ORGANISATIONS "," ORG ",1)
replace emp = subinstr( emp," ORGANISATION "," ORG ",1)
replace emp = subinstr( emp," ORGANIZATIONS "," ORG ",1)
replace emp = subinstr( emp," ORGANIZATION "," ORG ",1)
replace emp = subinstr( emp," ORGANIZZAZIONE "," ORG ",1)
replace emp = subinstr( emp," OSAKEYHTIO "," OY ",1)
replace emp = subinstr( emp," PHARMACEUTICALS "," PHARM ",1)
replace emp = subinstr( emp," PHARMACEUTICAL "," PHARM ",1)
replace emp = subinstr( emp," PHARMACEUTICA "," PHARM ",1)
replace emp = subinstr( emp," PHARMACEUTIQUES "," PHARM ",1)
replace emp = subinstr( emp," PHARMACEUTIQUE "," PHARM ",1)
replace emp = subinstr( emp," PHARMAZEUTIKA "," PHARM ",1)
replace emp = subinstr( emp," PHARMAZEUTISCHEN "," PHARM ",1)
replace emp = subinstr( emp," PHARMAZEUTISCHE "," PHARM ",1)
replace emp = subinstr( emp," PHARMAZEUTISCH "," PHARM ",1)
replace emp = subinstr( emp," PHARMAZIE "," PHARM ",1)
replace emp = subinstr( emp," PUBLIC LIMITED COMPANY "," PLC ",1)
replace emp = subinstr( emp," PRELUCRAREA "," PRELUC ",1)
replace emp = subinstr( emp," PRELUCRARE "," PRELUC ",1)
replace emp = subinstr( emp," PRODOTTI "," PROD ",1)
replace emp = subinstr( emp," PRODUCE "," PROD ",1)
replace emp = subinstr( emp," PRODUCTS "," PROD ",1)
replace emp = subinstr( emp," PRODUCT "," PROD ",1)
replace emp = subinstr( emp," PRODUCTAS "," PROD ",1)
replace emp = subinstr( emp," PRODUCTA "," PROD ",1)
replace emp = subinstr( emp," PRODUCTIE "," PROD ",1)
replace emp = subinstr( emp," PRODUCTOS "," PROD ",1)
replace emp = subinstr( emp," PRODUCTO "," PROD ",1)
replace emp = subinstr( emp," PRODUCTORES "," PROD ",1)
replace emp = subinstr( emp," PRODUITS "," PROD ",1)
replace emp = subinstr( emp," PRODUIT "," PROD ",1)
replace emp = subinstr( emp," PRODUKCJI "," PROD ",1)
replace emp = subinstr( emp," PRODUKTER "," PROD ",1)
replace emp = subinstr( emp," PRODUKTE "," PROD ",1)
replace emp = subinstr( emp," PRODUKT "," PROD ",1)
replace emp = subinstr( emp," PRODUSE "," PROD ",1)
replace emp = subinstr( emp," PRODUTOS "," PROD ",1)
replace emp = subinstr( emp," PRODUIT CHIMIQUES "," PROD CHIM ",1)
replace emp = subinstr( emp," PRODUIT CHIMIQUE "," PROD CHIM ",1)
replace emp = subinstr( emp," PRODUCTIONS "," PRODN ",1)
replace emp = subinstr( emp," PRODUCTION "," PRODN ",1)
replace emp = subinstr( emp," PRODUKTIONS "," PRODN ",1)
replace emp = subinstr( emp," PRODUKTION "," PRODN ",1)
replace emp = subinstr( emp," PRODUZIONI "," PRODN ",1)
replace emp = subinstr( emp," PROIECTARE "," PROI ",1)
replace emp = subinstr( emp," PROIECTARI "," PROI ",1)
replace emp = subinstr( emp," PRZEDSIEBIOSTWO "," PRZEDSIEB ",1)
replace emp = subinstr( emp," PRZEMYSLU "," PRZEYM ",1)
replace emp = subinstr( emp," PROPRIETARY "," PTY ",1)
replace emp = subinstr( emp," PERSONENVENNOOTSCHAP MET "," PVBA ",1)
replace emp = subinstr( emp," BEPERKTE AANSPRAKELIJKHEID "," PVBA ",1)
replace emp = subinstr( emp," REALISATIONS "," REAL ",1)
replace emp = subinstr( emp," REALISATION "," REAL ",1)
replace emp = subinstr( emp," RECHERCHES "," RECH ",1)
replace emp = subinstr( emp," RECHERCHE "," RECH ",1)
replace emp = subinstr( emp," RECHERCHES ET DEVELOPMENTS "," RECH & DEV ",1)
replace emp = subinstr( emp," RECHERCHE ET DEVELOPMENT "," RECH & DEV ",1)
replace emp = subinstr( emp," RECHERCHES ET DEVELOPPEMENTS "," RECH & DEV ",1)
replace emp = subinstr( emp," RECHERCHE ET DEVELOPPEMENT "," RECH & DEV ",1)
replace emp = subinstr( emp," RESEARCH & DEVELOPMENT "," RES & DEV ",1)
replace emp = subinstr( emp," RESEARCH AND DEVELOPMENT "," RES & DEV ",1)
replace emp = subinstr( emp," RESEARCH "," RES ",1)
replace emp = subinstr( emp," RIJKSUNIVERSITEIT "," RIJKSUNIV ",1)
replace emp = subinstr( emp," SECRETATY "," SECRETARY ",1)
replace emp = subinstr( emp," SECRETRY "," SECRETARY ",1)
replace emp = subinstr( emp," SECREATRY "," SECRETARY ",1)
replace emp = subinstr( emp," SOCIEDAD ANONIMA "," SA ",1)
replace emp = subinstr( emp," SOCIETE ANONYME DITE "," SA ",1)
replace emp = subinstr( emp," SOCIETE ANONYME "," SA ",1)
replace emp = subinstr( emp," SOCIETE A RESPONSABILITE LIMITEE "," SARL ",1)
replace emp = subinstr( emp," SOCIETE A RESPONSIBILITE LIMITEE "," SARL ",1)
replace emp = subinstr( emp," SOCIETA IN ACCOMANDITA SEMPLICE "," SAS ",1)
replace emp = subinstr( emp," SCHWEIZERISCHES "," SCHWEIZ ",1)
replace emp = subinstr( emp," SCHWEIZERISCHER "," SCHWEIZ ",1)
replace emp = subinstr( emp," SCHWEIZERISCHE "," SCHWEIZ ",1)
replace emp = subinstr( emp," SCHWEIZERISCH "," SCHWEIZ ",1)
replace emp = subinstr( emp," SCHWEIZER "," SCHWEIZ ",1)
replace emp = subinstr( emp," SCIENCES "," SCI ",1)
replace emp = subinstr( emp," SCIENCE "," SCI ",1)
replace emp = subinstr( emp," SCIENTIFICA "," SCI ",1)
replace emp = subinstr( emp," SCIENTIFIC "," SCI ",1)
replace emp = subinstr( emp," SCIENTIFIQUES "," SCI ",1)
replace emp = subinstr( emp," SCIENTIFIQUE "," SCI ",1)
replace emp = subinstr( emp," SHADAN HOJIN "," SH ",1)
replace emp = subinstr( emp," SIDERURGICAS "," SIDER ",1)
replace emp = subinstr( emp," SIDERURGICA "," SIDER ",1)
replace emp = subinstr( emp," SIDERURGIC "," SIDER ",1)
replace emp = subinstr( emp," SIDERURGIE "," SIDER ",1)
replace emp = subinstr( emp," SIDERURGIQUE "," SIDER ",1)
replace emp = subinstr( emp," SOCIETA IN NOME COLLECTIVO "," SNC ",1)
replace emp = subinstr( emp," SOCIETE EN NOM COLLECTIF "," SNC ",1)
replace emp = subinstr( emp," SOCIETE ALSACIENNE "," SOC ALSAC ",1)
replace emp = subinstr( emp," SOCIETE APPLICATION "," SOC APPL ",1)
replace emp = subinstr( emp," SOCIETA APPLICAZIONE "," SOC APPL ",1)
replace emp = subinstr( emp," SOCIETE AUXILIAIRE "," SOC AUX ",1)
replace emp = subinstr( emp," SOCIETE CHIMIQUE "," SOC CHIM ",1)
replace emp = subinstr( emp," SOCIEDAD CIVIL "," SOC CIV ",1)
replace emp = subinstr( emp," SOCIETE CIVILE "," SOC CIV ",1)
replace emp = subinstr( emp," SOCIETE COMMERCIALES "," SOC COMML ",1)
replace emp = subinstr( emp," SOCIETE COMMERCIALE "," SOC COMML ",1)
replace emp = subinstr( emp," SOCIEDAD ESPANOLA "," SOC ESPAN ",1)
replace emp = subinstr( emp," SOCIETE ETUDES "," SOC ETUD ",1)
replace emp = subinstr( emp," SOCIETE ETUDE "," SOC ETUD ",1)
replace emp = subinstr( emp," SOCIETE EXPLOITATION "," SOC EXPL ",1)
replace emp = subinstr( emp," SOCIETE GENERALE "," SOC GEN ",1)
replace emp = subinstr( emp," SOCIETE INDUSTRIELLES "," SOC IND ",1)
replace emp = subinstr( emp," SOCIETE INDUSTRIELLE "," SOC IND ",1)
replace emp = subinstr( emp," SOCIETE MECANIQUES "," SOC MEC ",1)
replace emp = subinstr( emp," SOCIETE MECANIQUE "," SOC MEC ",1)
replace emp = subinstr( emp," SOCIETE NATIONALE "," SOC NAT ",1)
replace emp = subinstr( emp," SOCIETE NOUVELLE "," SOC NOUV ",1)
replace emp = subinstr( emp," SOCIETE PARISIENNE "," SOC PARIS ",1)
replace emp = subinstr( emp," SOCIETE PARISIENN "," SOC PARIS ",1)
replace emp = subinstr( emp," SOCIETE PARISIEN "," SOC PARIS ",1)
replace emp = subinstr( emp," SOCIETE TECHNIQUES "," SOC TECH ",1)
replace emp = subinstr( emp," SOCIETE TECHNIQUE "," SOC TECH ",1)
replace emp = subinstr( emp," SDRUZENI PODNIKU "," SP ",1)
replace emp = subinstr( emp," SDRUZENI PODNIK "," SP ",1)
replace emp = subinstr( emp," SOCIETA PER AZIONI "," SPA ",1)
replace emp = subinstr( emp," SPITALUL "," SPITAL ",1)
replace emp = subinstr( emp," SOCIETE PRIVEE A RESPONSABILITE LIMITEE "," SPRL ",1)
replace emp = subinstr( emp," SOCIEDAD DE RESPONSABILIDAD LIMITADA "," SRL ",1)
replace emp = subinstr( emp," STIINTIFICA "," STIINT ",1)
replace emp = subinstr( emp," SUDDEUTSCHES "," SUDDEUT ",1)
replace emp = subinstr( emp," SUDDEUTSCHER "," SUDDEUT ",1)
replace emp = subinstr( emp," SUDDEUTSCHE "," SUDDEUT ",1)
replace emp = subinstr( emp," SUDDEUTSCH "," SUDDEUT ",1)
replace emp = subinstr( emp," SOCIEDADE "," SOC ",1)
replace emp = subinstr( emp," SOCIEDAD "," SOC ",1)
replace emp = subinstr( emp," SOCIETA "," SOC ",1)
replace emp = subinstr( emp," SOCIETE "," SOC ",1)
replace emp = subinstr( emp," SOCIETY "," SOC ",1)
replace emp = subinstr( emp," SA DITE "," SA ",1)
replace emp = subinstr( emp," TECHNICAL "," TECH ",1)
replace emp = subinstr( emp," TECHNICO "," TECH ",1)
replace emp = subinstr( emp," TECHNICZNY "," TECH ",1)
replace emp = subinstr( emp," TECHNIKAI "," TECH ",1)
replace emp = subinstr( emp," TECHNIKI "," TECH ",1)
replace emp = subinstr( emp," TECHNIK "," TECH ",1)
replace emp = subinstr( emp," TECHNIQUES "," TECH ",1)
replace emp = subinstr( emp," TECHNIQUE "," TECH ",1)
replace emp = subinstr( emp," TECHNISCHES "," TECH ",1)
replace emp = subinstr( emp," TECHNISCHE "," TECH ",1)
replace emp = subinstr( emp," TECHNISCH "," TECH ",1)
replace emp = subinstr( emp," TECHNOLOGY "," TECH ",1)
replace emp = subinstr( emp," TECHNOLOGIES "," TECH ",1)
replace emp = subinstr( emp," TELECOMMUNICATIONS "," TELECOM ",1)
replace emp = subinstr( emp," TELECOMMUNICACION "," TELECOM ",1)
replace emp = subinstr( emp," TELECOMMUNICATION "," TELECOM ",1)
replace emp = subinstr( emp," TELECOMMUNICAZIONI "," TELECOM ",1)
replace emp = subinstr( emp," TELECOMUNICAZIONI "," TELECOM ",1)
replace emp = subinstr( emp," TRUSTUL "," TRUST ",1)
replace emp = subinstr( emp," UNITED KINGDOM "," UK ",1)
replace emp = subinstr( emp," SECRETARY OF STATE FOR "," UK SEC FOR ",1)
replace emp = subinstr( emp," UNIVERSIDADE "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSIDAD "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITA DEGLI STUDI "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITAET "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITAIRE "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITAIR "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITATEA "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITEIT "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETAMI "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETAM "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETE "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETOM "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETOV "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETU "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETY "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITETA "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITAT "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITET "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITE "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITY "," UNIV ",1)
replace emp = subinstr( emp," UNIVERSITA "," UNIV ",1)
replace emp = subinstr( emp," UNIWERSYTET "," UNIV ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA ADMINISTRATOR "," US ADMIN ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA AS REPRESENTED BY THE ADMINISTRATOR "," US ADMIN ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA AS REPRESENTED BY THE DEPT "," US DEPT ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA AS REPRESENTED BY THE UNITED STATES DEPT "," US DEPT ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICAN AS REPRESENTED BY THE UNITED STATES DEPT "," US DEPT ",1)
replace emp = subinstr( emp," UNITED STATES GOVERNMENT AS REPRESENTED BY THE SECRETARY OF "," US SEC ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA REPRESENTED BY THE SECRETARY "," US SEC ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA AS REPRESENTED BY THE SECRETARY "," US SEC ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICAS AS REPRESENTED BY THE SECRETARY "," US SEC ",1)
replace emp = subinstr( emp," UNITES STATES OF AMERICA AS REPRESENTED BY THE SECRETARY "," US SEC ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA SECRETARY OF "," US SEC ",1)
replace emp = subinstr( emp," UNITED STATES OF AMERICA "," USA ",1)
replace emp = subinstr( emp," UNITED STATES "," USA ",1)
replace emp = subinstr( emp," UTILAJE "," UTIL ",1)
replace emp = subinstr( emp," UTILAJ "," UTIL ",1)
replace emp = subinstr( emp," UTILISATIONS VOLKSEIGENER BETRIEBE "," VEB ",1)
replace emp = subinstr( emp," UTILISATION VOLKSEIGENER BETRIEBE "," VEB ",1)
replace emp = subinstr( emp," VEB KOMBINAT "," VEB KOMB ",1)
replace emp = subinstr( emp," VEREENIGDE "," VER ",1)
replace emp = subinstr( emp," VEREINIGTES VEREINIGUNG "," VER ",1)
replace emp = subinstr( emp," VEREINIGTE VEREINIGUNG "," VER ",1)
replace emp = subinstr( emp," VEREIN "," VER ",1)
replace emp = subinstr( emp," VERENIGING "," VER ",1)
replace emp = subinstr( emp," VERWALTUNGEN "," VERW ",1)
replace emp = subinstr( emp," VERWALTUNGS "," VERW ",1)
replace emp = subinstr( emp," VERWERTUNGS "," VERW ",1)
replace emp = subinstr( emp," VERWALTUNGSGESELLSCHAFT "," VERW GES ",1)
replace emp = subinstr( emp," VYZK USTAV "," VU ",1)
replace emp = subinstr( emp," VYZKUMNY USTAV "," VU ",1)
replace emp = subinstr( emp," VYZKUMNYUSTAV "," VU ",1)
replace emp = subinstr( emp," VEREINIGUNG VOLKSEIGENER BETRIEBUNG "," VVB ",1)
replace emp = subinstr( emp," VYZK VYVOJOVY USTAV "," VVU ",1)
replace emp = subinstr( emp," VYZKUMNY VYVOJOVY USTAV "," VVU ",1)
replace emp = subinstr( emp," WERKZEUGMASCHINENKOMBINAT "," WERKZ MASCH KOMB ",1)
replace emp = subinstr( emp," WERKZEUGMASCHINENFABRIK "," WERKZ MASCHFAB ",1)
replace emp = subinstr( emp," WESTDEUTSCHES "," WESTDEUT ",1)
replace emp = subinstr( emp," WESTDEUTSCHER "," WESTDEUT ",1)
replace emp = subinstr( emp," WESTDEUTSCHE "," WESTDEUT ",1)
replace emp = subinstr( emp," WESTDEUTSCH "," WESTDEUT ",1)
replace emp = subinstr( emp," WISSENSCHAFTLICHE(S) "," WISS ",1)
replace emp = subinstr( emp," WISSENSCHAFTLICHES TECHNISCHES ZENTRUM "," WTZ ",1)
replace emp = subinstr( emp," YUGEN KAISHA "," YG YUGEN GAISHA ",1)
replace emp = subinstr( emp," YUUGEN GAISHA "," YG YUGEN GAISHA ",1)
replace emp = subinstr( emp," YUUGEN KAISHA "," YG YUGEN GAISHA ",1)
replace emp = subinstr( emp," YUUGEN KAISYA "," YG YUGEN GAISHA ",1)
replace emp = subinstr( emp," ZAVODU "," ZAVOD ",1)
replace emp = subinstr( emp," ZAVODY "," ZAVOD ",1)
replace emp = subinstr( emp," ZENTRALES "," ZENT ",1)
replace emp = subinstr( emp," ZENTRALE "," ZENT ",1)
replace emp = subinstr( emp," ZENTRALEN "," ZENT ",1)
replace emp = subinstr( emp," ZENTRALNA "," ZENT ",1)
replace emp = subinstr( emp," ZENTRUM "," ZENT ",1)
replace emp = subinstr( emp," ZENTRALINSTITUT "," ZENT INST ",1)
replace emp = subinstr( emp," ZENTRALLABORATORIUM "," ZENT LAB ",1)
replace emp = subinstr( emp," ZAIDAN HOJIN "," ZH ",1)
replace emp = subinstr( emp," ZAIDAN HOUJIN "," ZH ",1)
replace emp = subinstr( emp," LIMITED "," LTD ",1)
replace emp = subinstr( emp," LIMITADA "," LTDA ",1)
replace emp = subinstr( emp," SECRETARY "," SEC ",1)
replace emp = subinstr( emp, " LTD ", " ", 1)
replace emp = subinstr( emp, " CO LTD ", " ", 1)
replace emp = subinstr( emp, " TRADING LTD ", " ", 1)
*replace emp = subinstr( emp, " HLDGS ", " ", 1)
replace emp = subinstr( emp, " CORP ", " ", 1)
replace emp = subinstr( emp, " INTL ", " ", 1)
replace emp = subinstr( emp, " INC ", " ", 1)
replace emp = subinstr( emp, " PLC ", " ", 1)
replace emp = subinstr( emp, " SPA ", " ", 1)
replace emp = subinstr( emp, " CLA ", " ", 1)
replace emp = subinstr( emp, " LLP ", " ", 1)
replace emp = subinstr( emp, " LLC ", " ", 1)
replace emp = subinstr( emp, " AIS ", " ", 1)
replace emp = subinstr( emp, " INVESTMENTS ", " ", 1)
replace emp = subinstr( emp, " PARTNERSHIP ", " ", 1)
replace emp = subinstr( emp, " AND CO ", " ", 1)
replace emp = subinstr( emp, " CO ", " ", 1)
replace emp = subinstr( emp, " COS ", " ", 1)
replace emp = subinstr( emp, " CP ", " ", 1)
replace emp = subinstr( emp, " LP ", " ", 1)
replace emp = subinstr( emp, " BLSA ", " ", 1)
replace emp = subinstr( emp, " GROUP ", " ", 1)
replace emp = subinstr( emp, "(THE)", "", 1)
replace emp = subinstr(emp,"CONSOLIDATED "," ",30)
replace emp = subinstr(emp," ACCPTNCE "," ACCEPTANCE ",30)
replace emp = subinstr(emp," BANCORPORATION "," BANCORP ",30)
replace emp = subinstr(emp," BANCORPORTN "," BANCORP ",30)
replace emp = subinstr(emp," BANCRP "," BANCORP ",30)
replace emp = subinstr(emp," BNCSHRS "," BANCSHARES ",30)
replace emp = subinstr(emp," BRWG "," BREWING ",30)
replace emp = subinstr(emp," CHEVRONTEXACO "," CHEVRON TEXACO ",30)
replace emp = subinstr(emp," CHSE "," CHASE ",30)
replace emp = subinstr(emp," COMMN "," COMMUNICATION ",30)
replace emp = subinstr(emp," COMMUN "," COMMUNICATION ",30)
replace emp = subinstr(emp," COMMUNICATNS "," COMMUNICATION ",30)
replace emp = subinstr(emp," COMMUNICATIONS "," COMMUNICATION ",30)
replace emp = subinstr(emp," DPT STS "," DEPT STORES ",30)
replace emp = subinstr(emp," DPT "," DEPT ",30)
replace emp = subinstr(emp," ENRGY "," ENERGY ",30)
replace emp = subinstr(emp," FINL "," FINANCIAL ",30)
replace emp = subinstr(emp," FNCL "," FINANCIAL ",30)
replace emp = subinstr(emp," GRP "," GROUP ",30)
replace emp = subinstr(emp," HLDGS "," HOLDINGS ",30)
replace emp = subinstr(emp," HLDG "," HOLDING ",30)
replace emp = subinstr(emp," HLT NTWK "," HEALTH NETWORK ",30)
replace emp = subinstr(emp," HTLS RES "," HOTELS & RESORTS ",30)
replace emp = subinstr(emp," HLTH "," HEALTH ",30)
replace emp = subinstr(emp," INTRTECHNLGY "," INTERTECHNOLOGY ",30)
replace emp = subinstr(emp," JPMORGAN "," J P MORGAN ",30)
replace emp = subinstr(emp," MED OPTIC "," MEDICAL OPTICS ",30)
replace emp = subinstr(emp," MINNESOTA MINING AND MANUFACTURING COMPANY "," 3M COMPANY ",30)
replace emp = subinstr(emp," NAT RES "," NATURAL RESOURCES ",30)
replace emp = subinstr(emp," NETWRKS "," NETWORK ",30)
replace emp = subinstr(emp," PHARMACTICALS "," PHARM ",30)
replace emp = subinstr(emp," PHARMACT "," PHARM ",30)
replace emp = subinstr(emp," PPTYS TST "," PROPERTIES TRUST ",30)
replace emp = subinstr(emp," PPTY "," PROPERTY ",30)
replace emp = subinstr(emp," PROPERTY TR "," PROPERTY TRUST ",30)
replace emp = subinstr(emp," PAC RAILWY "," PACIFIC RAILWAY ",30)
replace emp = subinstr(emp," SEMICONDTR "," SEMICONDUCTOR ",30)
replace emp = subinstr(emp," SOLU "," SOLUTIONS ",30)
replace emp = subinstr(emp," ST & ALMN "," STEEL & ALUMINUM ",30)
replace emp = subinstr(emp," STD "," STANDARD ",30)
replace emp = subinstr(emp," TECHNOLGS "," TECH ",30)
replace emp = subinstr(emp," TECHNOL "," TECH ",30)
replace emp = subinstr(emp," TRANSPORTATN "," TRANSPORTATION ",30)
replace emp = subinstr(emp," ADVERTSG "," ADVERTISING ",30)	 /* JB */
replace emp = subinstr(emp," ADVNTGE "," ADVANTAGE ",30)	 /* JB */
replace emp = subinstr(emp," AIRLN "," AIRLINES ",30)	 /* JB */
replace emp = subinstr(emp," AIRLS "," AIRLINES ",30)	 /* JB */
replace emp = subinstr(emp," AM "," AMERICA ",30)	 /* JB */
replace emp = subinstr(emp," AMER "," AMERICAN ",30)	 /* JB */
replace emp = subinstr(emp," APPLIAN "," APPLIANCES ",30)	 /* JB */
replace emp = subinstr(emp," APPLICTN "," APPLICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," ARCHTCTS "," ARCHITECTS ",30)	 /* JB */
replace emp = subinstr(emp," ASSD "," ASSOCIATED ",30)	 /* JB */
replace emp = subinstr(emp," ASSOC "," ASSOCIATES ",30)	 /* JB */
replace emp = subinstr(emp," ASSOCS "," ASSOCIATES ",30)	 /* JB */
replace emp = subinstr(emp," ATOMC "," ATOMIC ",30)	 /* JB */
replace emp = subinstr(emp," BANCSH "," BANCSHARES ",30)	 /* JB */
replace emp = subinstr(emp," BANCSHR "," BANCSHARES ",30)	 /* JB */
replace emp = subinstr(emp," BCSHS "," BANCSHARES ",30)	 /* JB */
replace emp = subinstr(emp," BK "," BANK ",30)	 /* JB */
replace emp = subinstr(emp," BLDGS "," BUILDINGS ",30)	 /* JB */
replace emp = subinstr(emp," BROADCASTG "," BROADCASTING ",30)	 /* JB */
replace emp = subinstr(emp," BTLNG "," BOTTLING ",30)	 /* JB */
replace emp = subinstr(emp," CBLVISION "," CABLEVISION ",30)	 /* JB */
replace emp = subinstr(emp," CENTRS "," CENTERS ",30)	 /* JB */
replace emp = subinstr(emp," CHAMPNSHIP "," CHAMPIONSHIP ",30)	 /* JB */
replace emp = subinstr(emp," CMMNCTNS "," COMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," CNVRSION "," CONVERSION ",30)	 /* JB */
replace emp = subinstr(emp," COFF "," COFFEE ",30)	 /* JB */
replace emp = subinstr(emp," COMM "," COMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," COMMUN "," COMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," COMMUNCTN "," COMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," COMMUNICTNS "," COMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," COMP "," COMPUTERS ",30)	 /* JB */
replace emp = subinstr(emp," COMPUTR "," COMPUTER ",30)	 /* JB */
replace emp = subinstr(emp," CONFERENCG "," CONFERENCING ",30)	 /* JB */
replace emp = subinstr(emp," CONSTRN "," CONSTR ",30)	 /* JB */
replace emp = subinstr(emp," CONTL "," CONTINENTAL ",30)	 /* JB */
replace emp = subinstr(emp," CONTNT "," CONTINENTAL ",30)	 /* JB */
replace emp = subinstr(emp," CONTRL "," CONTROL ",30)	 /* JB */
replace emp = subinstr(emp," CONTRL "," CONTROL ",30)	 /* JB */
replace emp = subinstr(emp," CTR "," CENTER ",30)	 /* JB */
replace emp = subinstr(emp," CTRS "," CENTERS ",30)	 /* JB */
replace emp = subinstr(emp," CVRGS "," COVERINGS ",30)	 /* JB */
replace emp = subinstr(emp," DEV "," DEVELOPMENT ",30)	 /* JB */
replace emp = subinstr(emp," DEVL "," DEVELOPMENT ",30)	 /* JB */
replace emp = subinstr(emp," DEVLP "," DEVELOPMENT ",30)	 /* JB */
replace emp = subinstr(emp," DISTR "," DISTRIBUTION ",30)	 /* JB */
replace emp = subinstr(emp," DISTRIBUT "," DISTRIBUTION ",30)	 /* JB */
replace emp = subinstr(emp," DISTRIBUTN "," DISTRIBUTION ",30)	 /* JB */
replace emp = subinstr(emp," ELCTRNCS "," ELECTRONICS ",30)	 /* JB */
replace emp = subinstr(emp," ELECTR "," ELECTRONICS ",30)	 /* JB */
replace emp = subinstr(emp," ENGNRD "," ENGINEERED ",30)	 /* JB */
replace emp = subinstr(emp," ENMT "," ENTERTAINMENT ",30)	 /* JB */
replace emp = subinstr(emp," ENTERTAIN "," ENTERTAINMENT ",30)	 /* JB */
replace emp = subinstr(emp," ENTERTNMNT "," ENTERTAINMENT ",30)	 /* JB */
replace emp = subinstr(emp," ENTMNT "," ENTERTAINMENT ",30)	 /* JB */
replace emp = subinstr(emp," ENTMT "," ENTERTAINMENT ",30)	 /* JB */
replace emp = subinstr(emp," ENTRPR "," ENTERPRISES ",30)	 /* JB */
replace emp = subinstr(emp," ENTRPRISE "," ENTERPRISES ",30)	 /* JB */
replace emp = subinstr(emp," ENTRPRS "," ENTERPRISES ",30)	 /* JB */
replace emp = subinstr(emp," ENVIR "," ENVIRONMENTAL ",30)	 /* JB */
replace emp = subinstr(emp," ENVIRNMNTL "," ENVIRONMENTAL ",30)	 /* JB */
replace emp = subinstr(emp," ENVR "," ENVIRONMENTAL ",30)	 /* JB */
replace emp = subinstr(emp," EQUIPMT "," EQUIPMENT ",30)	 /* JB */
replace emp = subinstr(emp," EXCHG "," EXCHANGE ",30)	 /* JB */
replace emp = subinstr(emp," EXPLOR "," EXPLORATION ",30)	 /* JB */
replace emp = subinstr(emp," FNDG "," FUNDING ",30)	 /* JB */
replace emp = subinstr(emp," GLD "," GOLD ",30)	 /* JB */
replace emp = subinstr(emp," GP "," GROUP ",30)	 /* JB */
replace emp = subinstr(emp," HLDS "," HLDGS ",30)	 /* JB */
replace emp = subinstr(emp," HLTHCARE "," HEALTHCARE ",30)	 /* JB */
replace emp = subinstr(emp," HLTHCR "," HEALTHCARE ",30)	 /* JB */
replace emp = subinstr(emp," HOMEMDE "," HOMEMADE ",30)	 /* JB */
replace emp = subinstr(emp," HSPTL "," HOSPITAL ",30)	 /* JB */
replace emp = subinstr(emp," ILLUM "," ILLUMINATION ",30)	 /* JB */
replace emp = subinstr(emp," INDL "," INDUSTRIAL ",30)	 /* JB */
replace emp = subinstr(emp," INDPT "," INDEPENDENT ",30)	 /* JB */
replace emp = subinstr(emp," INDTY "," INDEMNITY ",30)	 /* JB */
replace emp = subinstr(emp," INFORMATN "," INFO ",30)	 /* JB */
replace emp = subinstr(emp," INSTNS "," INSTITUTIONS ",30)	 /* JB */
replace emp = subinstr(emp," INSTRUMEN "," INSTRUMENTS ",30)	 /* JB */
replace emp = subinstr(emp," INSTRUMNT "," INSTRUMENTS ",30)	 /* JB */
replace emp = subinstr(emp," INTEGRATRS "," INTEGRATORS ",30)	 /* JB */
replace emp = subinstr(emp," INTERNATIONL "," INT ",30)	 /* JB */
replace emp = subinstr(emp," INVS "," INVESTMENTS ",30)	 /* JB */
replace emp = subinstr(emp," INVT "," INVESTMENT ",30)	 /* JB */
replace emp = subinstr(emp," MANAGEMNT "," MANAGEMENT ",30)	 /* JB */
replace emp = subinstr(emp," MANAGMNT "," MANAGEMENT ",30)	 /* JB */
replace emp = subinstr(emp," MANHATN "," MANHATTAN ",30)	 /* JB */
replace emp = subinstr(emp," MANUF "," MFG ",30)	 /* JB */
replace emp = subinstr(emp," MDSE "," MERCHANDISING ",30)	 /* JB */
replace emp = subinstr(emp," MEASURMNT "," MEASUREMENT ",30)	 /* JB */
replace emp = subinstr(emp," MERCHNDSNG "," MERCHANDISING ",30)	 /* JB */
replace emp = subinstr(emp," MGMT "," MANAGEMENT ",30)	 /* JB */
replace emp = subinstr(emp," MGRS "," MANAGERS ",30)	 /* JB */
replace emp = subinstr(emp," MGT "," MANAGEMENT ",30)	 /* JB */
replace emp = subinstr(emp," MICROWAV "," MICROWAVE ",30)	 /* JB */
replace emp = subinstr(emp," MKTS "," MARKETS ",30)	 /* JB */
replace emp = subinstr(emp," MLTIMEDIA "," MULTIMEDIA ",30)	 /* JB */
replace emp = subinstr(emp," MTG "," MORTGAGE ",30)	 /* JB */
replace emp = subinstr(emp," MTNS "," MOUTAINS ",30)	 /* JB */
replace emp = subinstr(emp," MTRS "," MOTORS ",30)	 /* JB */
replace emp = subinstr(emp," NETWRK "," NETWORK ",30)	 /* JB */
replace emp = subinstr(emp," NOWEST "," NORTHWEST ",30)	 /* JB */
replace emp = subinstr(emp," NTWRK "," NETWORK ",30)	 /* JB */
replace emp = subinstr(emp," OFFSHRE "," OFFSHORE ",30)	 /* JB */
replace emp = subinstr(emp," ORGANIZTN "," ORG ",30)	 /* JB */
replace emp = subinstr(emp," PBLG "," PUBLISHING ",30)	 /* JB */
replace emp = subinstr(emp," PHARMACEUTICL "," PHARM ",30)	 /* JB */
replace emp = subinstr(emp," PLAST "," PLASTICS ",30)	 /* JB */
replace emp = subinstr(emp," PPTYS "," PROPERTIES ",30)	 /* JB */
replace emp = subinstr(emp," PRODS "," PROD ",30)	 /* JB */
replace emp = subinstr(emp," PRODTN "," PRODN ",30)	 /* JB */
replace emp = subinstr(emp," PRODUCTN "," PRODN ",30)	 /* JB */
replace emp = subinstr(emp," PRPANE "," PROPANE ",30)	 /* JB */
replace emp = subinstr(emp," PTS "," PARTS ",30)	 /* JB */
replace emp = subinstr(emp," PUBLISH "," PUBLISHING ",30)	 /* JB */
replace emp = subinstr(emp," PUBLSHING "," PUBLISHING ",30)	 /* JB */
replace emp = subinstr(emp," PUBN "," PUBLICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," PUBNS "," PUBLICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," PWR "," POWER ",30)	 /* JB */
replace emp = subinstr(emp," RAILRD "," RAILROAD ",30)	 /* JB */
replace emp = subinstr(emp," RECREATN "," RECREATION ",30)	 /* JB */
replace emp = subinstr(emp," RECYCL "," RECYCLING ",30)	 /* JB */
replace emp = subinstr(emp," REFIN "," REFINING ",30)	 /* JB */
replace emp = subinstr(emp," REFNG "," REFINING ",30)	 /* JB */
replace emp = subinstr(emp," RESTR "," RESTAURANT ",30)	 /* JB */
replace emp = subinstr(emp," RESTS "," RESTAURANTS ",30)	 /* JB */
replace emp = subinstr(emp," RETAILNG "," RETAILING ",30)	 /* JB */
replace emp = subinstr(emp," RLTY "," REALTY ",30)	 /* JB */
replace emp = subinstr(emp," RR "," RAILROAD ",30)	 /* JB */
replace emp = subinstr(emp," RSCH "," RESEARCH ",30)	 /* JB */
replace emp = subinstr(emp," RTNG "," RATING ",30)	 /* JB */
replace emp = subinstr(emp," SCIENTIF "," SCIENTIFIC ",30)	 /* JB */
replace emp = subinstr(emp," SERV "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," SLTNS "," SOLUTIONS ",30)	 /* JB */
replace emp = subinstr(emp," SOFTWRE "," SOFTWARE ",30)	 /* JB */
replace emp = subinstr(emp," SOLTNS "," SOLUTIONS ",30)	 /* JB */
replace emp = subinstr(emp," SOLUT "," SOLUTIONS ",30)	 /* JB */
replace emp = subinstr(emp," SRVC "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," SRVCS "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," STEAKHSE "," STEAKHOUSE ",30)	 /* JB */
replace emp = subinstr(emp," STHWST "," SOUTHWEST ",30)	 /* JB */
replace emp = subinstr(emp," STL "," STEEL ",30)	 /* JB */
replace emp = subinstr(emp," STRS "," STORES ",30)	 /* JB */
replace emp = subinstr(emp," SUP "," SUPPLY ",30)	 /* JB */
replace emp = subinstr(emp," SUPERMKTS "," SUPERMARKETS ",30)	 /* JB */
replace emp = subinstr(emp," SUPP "," SUPPLIES ",30)	 /* JB */
replace emp = subinstr(emp," SURVYS "," SURVEYS ",30)	 /* JB */
replace emp = subinstr(emp," SVC "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," SVCS "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," SVSC "," SERVICES ",30)	 /* JB */
replace emp = subinstr(emp," SYS "," SYSTEMS ",30)	 /* JB */
replace emp = subinstr(emp," SYSTM "," SYSTEMS ",30)	 /* JB */
replace emp = subinstr(emp," TCHNLGY "," TECH ",30)	 /* JB */
replace emp = subinstr(emp," TECHNGS "," TECHNOLOGIES ",30)	 /* JB */
replace emp = subinstr(emp," TECHNL "," TECH ",30)	 /* JB */
replace emp = subinstr(emp," TECHNLGIES "," TECHNOLOGIES ",30)	 /* JB */
replace emp = subinstr(emp," TEL "," TELEPHONE ",30)	 /* JB */
replace emp = subinstr(emp," TELE-COMM "," TELECOMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," TELE-COMMUN "," TELECOMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," TELECOMMS "," TELECOMMUNICATIONS ",30)	 /* JB */
replace emp = subinstr(emp," TELECONFERENC "," TELECONFERENCING ",30)	 /* JB */
replace emp = subinstr(emp," TELEG "," TELEGRAPH ",30)	 /* JB */
replace emp = subinstr(emp," TELEGR "," TELEGRAPH ",30)	 /* JB */
replace emp = subinstr(emp," TELVSN "," TELEVISION ",30)	 /* JB */
replace emp = subinstr(emp," TR "," TRUST ",30)	 /* JB */
replace emp = subinstr(emp," TRANSN "," TRANSPORTATION ",30)	 /* JB */
replace emp = subinstr(emp," TRANSPORTN "," TRANSPORTATION ",30)	 /* JB */
replace emp = subinstr(emp," TRNSACTN "," TRANSACTION ",30)	 /* JB */
replace emp = subinstr(emp," UTD "," UNITED ",30)	 /* JB */
replace emp = subinstr(emp," WSTN "," WESTERN ",30)	 /* JB */
replace emp = subinstr(emp," WTR "," WATER ",30)	 /* JB */
*  replace emp=" U.S. PHILIPS CORPORATION " if trim(emp)=="NORTH AMERICAN PHILIPS CORP"
replace emp=" A. L. WILLIAMS CORP. " if trim(emp)=="WILLIAMS (A.L.) CORP"
replace emp=" B. F. GOODRICH CO. " if trim(emp)=="GOODRICH CORP"
replace emp=" BELL + HOWELL COMPANY " if trim(emp)=="BELL & HOWELL OPERATING CO"
replace emp=" BENDIX CORPORATION(NOW ALLIED-SIGNAL INC.) " if trim(emp)=="BENDIX CORP"
replace emp=" BORG-WARNER CORPORATION " if trim(emp)=="BORGWARNER INC"
replace emp=" CHRYSLER MOTORS CORPORATION " if trim(emp)=="CHRYSLER CORP"
replace emp=" CISCO TECHNOLOGY, INC. " if trim(emp)=="CISCO SYSTEMS INC"
replace emp=" DELL PRODUCTS, L.P. " if trim(emp)=="DELL INC"
replace emp=" DELPHI TECHNOLOGIES, INC. " if trim(emp)=="DELPHI CORP"
replace emp=" E. I. DU PONT DE NEMOURS AND COMPANY " if trim(emp)=="DU PONT (E I) DE NEMOURS"
replace emp=" E. R. SQUIBB + SONS, INC. " if trim(emp)=="SQUIBB CORP"
replace emp=" ELI LILLY AND COMPANY " if trim(emp)=="LILLY (ELI) & CO"
replace emp=" G. D. SEARLE & CO. " if trim(emp)=="SEARLE (G.D.) & CO"
replace emp=" MINNESOTA MINING AND MANUFACTURING COMPANY " if trim(emp)=="3M CO"
replace emp=" OWENS-CORNING FIBERGLAS CORPORATION " if trim(emp)=="OWENS CORNING"
replace emp=" SCHLUMBERGER TECHNOLOGY CORPORATION " if trim(emp)=="SCHLUMBERGER LTD"
replace emp=" SCI-MED LIFE SYSTEMS, INC. " if trim(emp)=="SICMED LIFE SYSTEMS"
replace emp=" TDK CORPORATION " if trim(emp)=="TDK CORP"
replace emp=" UNITED STATES SURGICAL CORPORATION " if trim(emp)=="U S SURGICAL CORP"
replace emp=" W. R. GRACE & CO. " if trim(emp)=="GRACE (W R) & CO"
replace emp=" WESTINGHOUSE ELECTRIC CORP. " if trim(emp)=="WESTINGHOUSE ELEC"
replace emp = strlower(emp)

*trim back down again
replace emp = stritrim(emp)
replace emp = strtrim(emp)

*Fix up punctuation
replace emp=subinstr(emp, ",", "", .)

*Run regex replace a sufficient number of times to get all characters. only need to do 130 because anything longer than 128 was truncated
forvalues i = 1/130 {
	replace emp = regexr(emp, "\\.", "")
}
replace emp=subinstr(emp, "/", "", .)
replace emp=subinstr(emp, "'", "", .)
replace emp=subinstr(emp, "www", "", .)
*Remove org at the end of names
replace emp = substr(emp, 1, length(emp) - 2) if substr(emp, length(emp)-2, 3) == "org"
replace emp=subinstr(emp, "cntr", "center", .)
replace emp=subinstr(emp, "theatre", "theater", .)

replace emp = "schwan food" if emp == "schwans company"
replace emp = "schwan food" if ustrregexm(emp, "schwan food")
replace emp = "schwan food" if emp == "schwans shared services"

replace emp = "schneider national" if emp == "schneider nation"
replace emp = "schneider national" if emp == "schneider truck driving j"
replace emp = "schneider national" if emp == "schneider"

replace emp = "schneider legal group" if emp == "schneider law"

replace emp = "schneider company" if emp == "schneider com"

*Big firm fixes
replace emp = "walmart" if emp == "wal mart"
replace emp = "amazon" if emp == "amazon com"
replace emp = "jpmorgan chase" if emp == "jp morgan chase"
replace emp = "the home depot" if emp == "home depot"
replace emp = "united parcel service" if emp == "ups"
replace emp = "lowes" if emp == "lowes home improvement"

*Remove repeated internal whitespace (equivalent of str_squish)
replace emp = stritrim(emp)

*Isaac's RegEx addition:
*Set all-numeric and date format emp names to missing
count if ustrregexm(emp, "^[0-9 \/\-]+$")
replace emp = "" if ustrregexm(emp, "^[0-9 \/\-]+$")

********************************************************************************

* check for some problematic firm names
tab employer_original if ustrregexm(employer_original, "&Resources")
tab employer_original if ustrregexm(employer_original, "' C OLDJUSTICE")
tab employer_original if ustrregexm(employer_original, "Selected Clients/Partnerships AT&T MA Mutual Hanover Insurance Mercantile Bank & Trust Insurance Information Institute Commerce")
tab employer_original if ustrregexm(employer_original, "2005 /p p /p p strong SKILLS LEARNED")
tab employer_original if ustrregexm(employer_original, "276' valign='top' p ' Develop ")
tab employer_original if ustrregexm(employer_original, "'CPR and First Aid'")
tab employer_original if ustrregexm(employer_original, "'Jennifer Aniston' for")
tab emp if ustrregexm(employer_original, "&Resources")
tab emp if ustrregexm(employer_original, "' C OLDJUSTICE")
tab emp if ustrregexm(employer_original, "Selected Clients/Partnerships AT&T MA Mutual Hanover Insurance Mercantile Bank & Trust Insurance Information Institute Commerce")
tab emp if ustrregexm(employer_original, "2005 /p p /p p strong SKILLS LEARNED")
tab emp if ustrregexm(employer_original, "276' valign='top' p ' Develop ")
tab emp if ustrregexm(employer_original, "'CPR and First Aid'")
tab emp if ustrregexm(employer_original, "'Jennifer Aniston' for")

summarize N

*Another set of edits with the hospital cleaning code

*Remove spaces and go to all capital letters for best match
*Save a version of employer_original with only minor edits for hospital names
generate employer = employer_original
replace employer = subinstr(employer," ","",.)
replace employer = upper(employer)

*start making corrections using the emp2 variable
generate emp2 = employer

*Manual changes
replace emp2 = "ABBEVILLEGENERALHOSPITAL" if emp2=="ABBEVILLEGENERAL"
replace emp2 = "ABILENEREGIONALMEDICALCENTER" if strpos(emp2,"ABILENEREGIONALMEDICAL")>0
replace emp2 = "ABINGTONHOSPITAL" if strpos(emp2,"ABINGTONHOSPITAL")>0
replace emp2 = "ABINGTONHOSPITAL" if strpos(emp2,"ABINGTONMEMORIALHOSPITAL")>0
replace emp2 = "ABRAHAMLINCOLNPRESIDENTIALLIBRARY" if strpos(emp2,"ABRAHAMLINCOLNPRESIDENTIALLIBRARY")>0
replace emp2 = "TENETHEALTHCARECORPORATION" if strpos(emp2,"ABRAZO")>0
replace emp2 = "ACADIAHOSPITAL" if strpos(emp2,"ACADIAHOSPITAL")>0
replace emp2 = "ACADIAHEALTHCARE" if strpos(emp2,"ACADIAHEALHCARE")>0
replace emp2 = subinstr(emp2,"LLC","",.) if substr(emp2, -3, 3)=="LLC"
replace emp2 = "ADAIRCOUNTYMEMORIALHOSPITAL" if strpos(emp2,"ADAIRCOUNTYMEMORIALHOSPITAL")>0
replace emp2 = "ADAIRCOUNTYMEMORIALHOSPITAL" if strpos(emp2,"ADAIRCOUNTYMEMORIALHOSPITAL")>0
replace emp2 = "ADVENTISTHEALTHPHYSICIANSERVICES" if strpos(emp2,"ADVENTISTHEALTHPHYSICIAN")>0
replace emp2 = "ADVENTISTHEALTH" if emp2=="ADVENTISTHEALTHSOUTHERNCALIFORNIA"|emp2=="ADVENTISTHEALTHSOUTHERNCALIFORNIANETWORK"|emp2=="ADVENTISTHEALTHCENTRALCALIFORNIAREGION"|emp2=="ADVENTISTHEALTHCENTRALVALLEYNETWORK"
replace emp2 = "ADVENTISTHOMEASSISTANCE" if strpos(emp2,"ADVENTISTHOMEASSISTANCE")>0
replace emp2 = "ADVENTISTHINSDALEHOSPITAL" if strpos(emp2,"ADVENTISTHINSDALEHOSPITAL")>0
replace emp2 = "ADVENTISTHOMEHEALTH" if strpos(emp2,"ADVENTISTHOMEHLTH")>0|strpos(emp2,"ADVENTISTHOMEHEALTH")>0
replace emp2 = "ADVOCATEHEALTHCARE" if strpos(emp2,"ADVOCATECHILDRENS")>0|emp2=="ADVOCATECHRISTMEDICALCENTER"|emp2=="ADVOCATECHRISTMEDICALCENTER"|emp2=="ADVOCATEGOODSAMARITANHOSPITAL"|strpos(emp2,"ADVOCATEHEALTH")>0|emp2=="ADVOCATEILLINOISMEDICALCENTER"|emp2=="ADVOCATEILLINOISMASONICMEDICALCENTER"|emp2=="ADVOCATELUTHERANGENERALHOSPITAL"|emp2=="ADVOCATETRINITYHOSPITAL"
replace emp2 = "ADVOCATEHOMECARE" if strpos(emp2,"ADVOCATEHOMECARE")>0
replace emp2 = "ADVOCATEHOMESPECIALTYCARE" if strpos(emp2,"ADVOCATEHOMESPECIALTYCARE")>0
replace emp2 = "ADVOCATEMEDICALGROUP" if strpos(emp2,"ADVOCATEMEDICALGROUP")>0
replace emp2 = "ADVOCATEMERCHANTSOLUTIONS" if strpos(emp2,"ADVOCATEMERCHANTSOLUTIO")>0
replace emp2 = "ADVOCATERX" if strpos(emp2,"ADVOCATERX")>0
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="AIKENREGIONALMEDICALCENTERS"
replace emp2 = "CHILDRENSHOSPITALMEDICALCENTEROFAKRON" if strpos(emp2,"AKRON")>0 & strpos(emp2,"CHILD")>0 & strpos(emp2,"HOSPITAL")>0
replace emp2 = "THECLEVELANDCLINICHEALTHSYSTEM" if emp2=="AKRONGENERALMEDICALCENTER"
replace emp2 = "CAROLINASHEALTHCARESYSTEM" if emp2=="ALAMANCEREGIONALMEDICALCENTER"
replace emp2 = "HCA" if emp2=="ALASKAREGIONALHOSPITAL"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="ALBANYAREAHOSPITALANDMEDICALCENTER"
replace emp2 = "ALBANYMEDICALCENTER" if strpos(emp2,"ALBANYMEDICALCENTER")>0
replace emp2 = "TRINITYHEALTH" if strpos(emp2,"ALBANYMEMORIALHOSPITAL")>0
replace emp2 = "NEMOURSCHILDRENSHEALTHSYSTEM" if emp2=="ALFREDIDUPONTCHILDRENSHOSPITAL"
replace emp2 = "ALLEGHENYHEALTHNETWORK" if strpos(emp2,"ALLEGHENYGENERALHOSPITAL")>0|strpos(emp2,"ALLEGHENYVALLEYHOSPITAL")>0
replace emp2 = "HCA" if strpos(emp2,"ALLENCOUNTYREGIONALHOSPITAL")>0
replace emp2 = "NEWYORKPRESBYTERIANHEALTHCARESYSTEM" if strpos(emp2,"ALLENHOSPITAL")==1
replace emp2 = subinstr(emp2,"SYSTEMAMERICA","SYSTEM",.)
replace emp2 = subinstr(emp2,"SYSTEMUSA","SYSTEM",.)
replace emp2 = subinstr(emp2,"SYSTEMUS","SYSTEM",.)
replace emp2 = subinstr(emp2,"HOSPITALAMERICA","HOSPITAL",.)
replace emp2 = subinstr(emp2,"HOSPITALUSA","HOSPITAL",.)
replace emp2 = subinstr(emp2,"HOSPITALUS","HOSPITAL",.)
replace emp2 = subinstr(emp2,"SYSTEMSAMERICA","SYSTEM",.)
replace emp2 = subinstr(emp2,"SYSTEMSUSA","SYSTEM",.)
replace emp2 = subinstr(emp2,"SYSTEMSUS","SYSTEM",.)
replace emp2 = subinstr(emp2,"CENTERAMERICA","CENTER",.)
replace emp2 = subinstr(emp2,"CENTERUSA","CENTER",.)
replace emp2 = subinstr(emp2,"CENTERUS","CENTER",.)
replace emp2 = subinstr(emp2,"HEALTHSYSTEM","HEALTHSYSTEMS",.) if substr(emp2,-13,13)=="HEALTHSYSTEMS"
replace emp2 = subinstr(emp2,"COMMHOSPITAL","COMMUNITYHOSPITAL",.)
replace emp2 = subinstr(emp2,"CHILDRENACS","CHILDRENS",.)
replace emp2 = subinstr(emp2,"MEDCTR","MEDICALCENTER",.)
replace emp2 = subinstr(emp2,"HLTHCTR","HEALTHCENTER",.)

* *Merge based on hospital name alone
* merge m:1 employer using "${project_share}/Input/hospital_names/hospital_linkage_2016_noduplicatenames.dta", gen(_m_hosp)

* summarize N

* drop if _m_hosp==2

* summarize N

* replace health_sys_name = subinstr(emp2,"HEALTHSYSTEM","HEALTHSYSTEMS",.) if substr(emp2,-13,13)=="HEALTHSYSTEMS"
* replace emp2 = health_sys_name if _m_hosp==3
keep employer_original emp employer emp2 N

summarize N

*Continue with manual adjustments
replace emp2 = "ALLIANCEHEALTHCARESERVICES" if strpos(emp2,"ALLIANCEHEALTHCARESERVI")>0
replace emp2 = "ALLIANCEHEALTHCARESOLUTIONS" if strpos(emp2,"ALLIANCEHEALTHCARESOLUT")>0
replace emp2 = "ALLIANCEHOMEHEALTH" if strpos(emp2,"ALLIANCEHOMEHEALTH")==1
replace emp2 = "SUTTERHEALTH" if strpos(emp2,"ALTABATES")==1
replace emp2 = "PRIMEHEALTHCARESERVICES" if emp2=="ALVARADOHOSPITAL"
replace emp2 = "ANDERSONHOSPITAL" if strpos(emp2,"ANDERSONHOSPITAL")==1
replace emp2 = "ANNANDROBERTHLURIECHILDRENSHOSPITAL" if strpos(emp2,"ANN")==1 & strpos(emp2,"LURIE")>0 & strpos(emp2,"CHILD")>0
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="ARBOURFULLERHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"ARBOURHRI")==1
replace emp2 = "JEFFERSONHEALTH" if emp2=="ARIAJEFFERSONHEALTH"|strpos(emp2,"ARIAHEALTH")==1
replace emp2 = "ARKANSASCHILDRENSHOSPITAL" if strpos(emp2,"ARKANSAS")>0 & strpos(emp2,"CHILD")>0 & strpos(emp2,"HOSPITAL")>0
replace emp2 = "DIGNITYHEALTH" if strpos(emp2,"ARROYOGRANDECOMMUNITYHOSPIT")==1
replace emp2 = "ASANTEHEALTHSYSTEM" if strpos(emp2,"ASANTEHEALTHSYSTEM")==1|strpos(emp2,"ASANTEROGUE")==1|strpos(emp2,"ASANTETHREERIVERS")==1
replace emp2 = "MISSIONHEALTHSYSTEM" if emp2=="ASHEVILLESPECIALITYHOSPITAL"
replace emp2 = "HUNTSVILLEHOSPITALHEALTHSYSTEM" if emp2=="ATHENSLIMESTONEHOSPITAL"
replace emp2 = "BASSETTHEALTHCARENETWORK" if emp2 =="AURELIAOSBORNFOXMEMORIAL"
replace emp2 = "HCA" if emp2=="AVENTURAHOSPITAL26MEDICALCENTER"|emp2=="AVENTURAHOSPITALANDMEDIC"|emp2=="AVENTURAHOSPITAL"
replace emp2 = "AVERAHEALTH" if emp2=="AVERASTLUKESHOSPITAL"
replace emp2 = "BANNERHEALTH" if emp2=="BANNERUNIVERSITYMEDICALCENTER"|emp2=="BANNERUNIVERSITYMEDICALCENTERPHOENIX"|emp2=="BANNERBAYWOODHEARTHOSPITAL"|emp2=="BANNERDELEWEBBMEDICALCENTER"
replace emp2 = "GREENVILLEHEALTHSYSTEM" if strpos(emp2,"BAPTISTEASLEY")==1|strpos(emp2,"BAPTISTEASELY")==1
replace emp2 = "BAPTISTHEALTH" if strpos(emp2,"BAPTISTHEALTH")==1
replace emp2 = "BAPTISTMEMORIALHEALTHCARE" if emp2=="BAPTISTMEMORIALHEALTHC"|strpos(emp2,"BAPTISTMEMORIALHOSPITAL")==1
replace emp2 = "UPPERPENINSULAHEALTHCARESOLUTIONS" if emp2=="BARAGACOUNTYMEMORIALHOSPIITAL"
replace emp2 = "RWJBARNABASHEALTH" if strpos(emp2,"BARNABASHEALTH")>0
replace emp2 = "BJCHEALTHCARE" if strpos(emp2,"BARNESJEWISHHEALTH")==1|strpos(emp2,"BARNESJEWISHWEST")==1
replace emp2 = "UNIVERSITYOFMIAMIHEALTHSYSTEM" if strpos(emp2,"BASCOMPALMEREYE")==1
replace emp2 = "IASISHEALTHCARECORPORATION" if strpos(emp2,"BASINHEALTHCARE")==1
replace emp2 = "CARILIONCLINIC" if emp2=="BATHCOUNTYCOMMUNITYHOSPITAL"
replace emp2 = "BATONROUGEGENERALHEALTHSYSTEM" if emp2=="BATONROUGEGENERAL"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2 =="BAYFRONTHEALTH"
replace emp2 = "BAYLORSCOTTANDWHITEHEALTH" if strpos(emp2,"BAYLORALLSAINTS")==1|emp2=="BAYLORALLSANITSMEDICALCENT"|strpos(emp2,"BAYLOREMERGENCYMEDICAL")==1|emp2=="BAYLORINSTITUTEFORREHABILITATIONAMERICA"|strpos(emp2,"BAYLORJACKANDJANEHAMILTON")==1|strpos(emp2,"BAYLORMEDICALCENTER")==1|emp2=="BAYLORSCOTTANDWHITE"|emp2=="BAYLORSCOTTANDWHITEHEAL"|emp2=="BAYLORSCOTTANDWHITEHEALTHE"|emp2=="BAYLORSURGICALHOSPITALATFORT"|emp2=="BAYLORUNIVERSITYMEDICALCENTERATDALLAS"
replace emp2 = "CAREPOINTHEALTH" if emp2=="BAYONNEMEDICALCENTERUSA"
replace emp2 = "APPALACHIANREGIONALHEALTHCARE" if emp2=="BECKLEYARH"
replace emp2 = "BEEBEHEALTHCARE" if strpos(emp2,"BEEBEHEALTHCARE")==1
replace emp2 = "BELLINHEALTHSYSTEMS" if emp2=="BELLINHEALTH"|emp2=="BELLINHEALTHUS"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="BELLVILLESTJOSEPHHEALTHCENTERBELLEVILLE"
replace emp2 = "BELOITHEALTHSYSTEMS" if strpos(emp2,"BELOITHEALTHSYSTEM")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="BENCHMARKBEHAVIORALHEALTH"
replace emp2 = "BERGERHEALTHSYSTEM" if strpos(emp2,"BERGERHEALTHSYSTEM")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2 =="BERWICKHOSPITAL"
replace emp2 = "BETHISRAELDEACONESSMEDICALCENTER" if strpos(emp2,"BETHISRAELDEACONESSMED")==1|emp2=="BETHISRAELDEACONESS"|strpos(emp2,"BETHISRAELDEACONESSHEALTH")==1|strpos(emp2,"BETHISRAELMEDICALCENTER")==1|emp2=="BETHISRAELDEACONESSPLYMOUTH"
replace emp2 = "HCA" if emp2=="BLAKEMEDICALCENTERAMERICA"|emp2=="BLAKEMEDICALCENTERUS"
replace emp2 = "BLESSINGHEALTHSYSTEM" if emp2=="BLESSINGHOSPITALQUINCYIL"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"BLUEFIELDREGIONALMEDICA")==1
replace emp2 = "BONSECOURSHEALTHSYSTEM" if emp2=="BONSECOURSBALTIMOREHOSPITAL"|emp2=="BONSECOURSDEPAULMEDICALCENTER"|emp2=="BONSECOURSHOSPITAL"|emp2=="BONSECOURSHS"|emp2=="BONSECOURSHOSP"|emp2=="BONSECOURSMARYIMMACULATEHOSPITAL"|emp2=="BONSECOURSNEWYORKHEALTHSYSTEM"|emp2=="BONSECOURSSTFRANCISHEALTHSYSTEM"
replace emp2 = "TRINITYHEALTH" if emp2=="BOONECOUNTYHOSPITALAMERICA"
replace emp2 = "ASCENSIONHEALTH" if strpos(emp2,"BORGESSLEEMEMORIAL")==1|strpos(emp2,"BORGESSLEEMEDICALCENTER")==1|emp2=="BORGESSPIPPHOSPITAL"|emp2=="BORGESSHEALTH"
replace emp2 = "BOSTONCHILDRENSHOSPITAL" if strpos(emp2,"BOSTONCHILDRENSHOSPITAL")==1
replace emp2 = "HCA" if emp2=="BRANDONREGIONALHOSPITALBRANDON"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"BRANDYWINEHOSPITAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="BRENTWOODBEHAVIORALHEALTHOFMISSISSIPPI"
replace emp2 = "CENTRALMAINEHEALTHCARE" if emp2=="BRIDGTONHOSPITA"
replace emp2 = "PARTNERSHEALTHCARESYSTEM" if strpos(emp2,"BRIGHAMANDWOMEN")==1
replace emp2 = "BRONSONHEALTHCARE" if emp2=="BRONSONBATTLECREEKHOSPITAL"|strpos(emp2,"BRONSONHEALTHCARE")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"BROOKEGLEN")==1
replace emp2 = "LONGISLANDHEALTHNETWORK" if emp2=="BROOKHAVENMEMORIALHOSPITAL"|emp2=="BROOKHAVENMHC"
replace emp2 = "BROOKHAVENNATIONALLABORATORY" if strpos(emp2,"BROOKHAVENNATIONALLAB")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"BROWNWOODREGIONALMEDICALCENTER")==1
replace emp2 = "BRYANHEALTH" if emp2=="BRYANMEDICAL"
replace emp2 = "CARENEWENGLANDHEALTHSYSTEM" if strpos(emp2,"CARENEWENGLAND")==1
replace emp2 = "DIGNITYHEALTH" if emp2=="CALIFORNIAHOSPITALMEDCENTER"|emp2=="CALIFORNIAHOSPITALMEDICAL"|emp2=="CALIFORNIAHOSPITALMEDICALCTR"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="CALLAWAYDISTRICTHOSPITALANDMEDICALCLINICS"
replace emp2 = "ARCHCARE" if strpos(emp2,"CALVARYHOSPITAL")==1
replace emp2 = "WESTVIRGINIAUNITEDHEALTHSYSTEM" if strpos(emp2,"CAMDENCLARKMEDICALCENTERSTJOSEPHSCAMPUS")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="CANYONRIDGEHOSPITALUNIVERSALHEALTHSERVICES"
replace emp2 = "CAREPOINTHEALTH" if strpos(emp2,"CAREPOINTHEALTH")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="CARLISLEREGIONALMEDICAL"|emp2=="CARLISLEREGIONALHOSPITAL"
replace emp2 = "RCCHHEALTHCAREPARTNERS" if emp2=="CAROLINAPINESMEDICALCENTER"
replace emp2 = "CAROLINASHEALTHCARESYSTEM" if emp2=="CAROLINASHEALTHCARE"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="CAROLINASHOSPITALSYSTEMAMERICA"|strpos(emp2,"CAROLINASHOSPITALSYSTEMUS")==1
replace emp2 = "CARSONTAHOEHEALTH" if strpos(emp2,"CARSONTAHOEHEALTH")==1|strpos(emp2,"CARSONTAHOEREGIONAL")==1
replace emp2 = "HCA" if emp2=="CASSREGIONAL"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"CEDARPARKREGIONAL")==1
replace emp2 = "CENTEGRAHEALTHSYSTEMS" if strpos(emp2,"CENTEGRA")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="CENTENNIALHILLSHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"UNIVERSALHEALTHSERVICES")>0
replace emp2 = "PRIMEHEALTHCARESERVICES" if strpos(emp2,"CENTINELAHOSPITALMEDICALCENTER")==1
replace emp2 = "REGIONALHEALTH" if emp2=="CHADRONCOMMUNITIYHOSPITALANDHEALTHSERVICES"|emp2=="CHADRONCOMMUNITYHOSPITAL"|emp2=="CHADRONCOMMUNITYHOSPITALANDHEALTH"
replace emp2 = "DIGNITYHEALTH" if strpos(emp2,"CHANDLERREGIONAL")==1
replace emp2 = "EASTERNMAINEHEALTHCARESYSTEMS" if strpos(emp2,"CHARLESADEANMEMORIAL")==1
replace emp2 = subinstr(emp2,"REHABILITATION","REHAB",.)
replace emp2 = subinstr(emp2,"REHABILTATION","REHAB",.)
replace emp2 = "CATHOLICHEALTHINITIATIVES" if strpos(emp2,"CHIHEALTH")==1|strpos(emp2,"CHIMERCY")==1|strpos(emp2,"CHISTJOSEPHHEALTH")==1|strpos(emp2,"CHISTLUKE")==1|strpos(emp2,"CHISTVINCENT")==1
replace emp2 = "AVERAHEALTH" if strpos(emp2,"CHIPPEWACOUNTYMONTEVIDEOHOS")==1
replace emp2 = "THECHRISTHOSPITALHEALTHNETWORK" if emp2 == "CHRISTHOSPITALOHIO" | emp2 == "CHRISTHOSPITALUS"
replace emp2 = "CHRISTUSHEALTH" if strpos(emp2,"CHRISTUSDUBUIS")==1|emp2=="CHRISTUSHEALTHCHRISTUSHEALTH"|emp2=="CHRISTUSHIGHLANDMEDICALCENTER"|strpos(emp2,"CHRISTUSHOSPITAL")==1|strpos(emp2,"CHRISTUSSANTAROSA")==1|strpos(emp2,"CHRISTUSSPOHN")==1|strpos(emp2,"CHRISTUSSTMICHAEL")==1|strpos(emp2,"CHRISTUSSTPATRICK")==1|strpos(emp2,"CHRISTUSSTVINCENT")==1|strpos(emp2,"CHRISTUSTRINITYMOTHER")==1
replace emp2 = "HCA" if emp2=="CITRUSMEMORIALHEALTHSYSTEMS"|emp2=="CITRUSMEMORIALHEALTHSYSTEMUS"|emp2=="CITRUSMEMRIALHEALTHSYSTEM"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="CLARIONPSYCHIATRIC"
replace emp2 = "LIFEPOINTHEALTH" if emp2 == "CLARKMEMORIAL"
replace emp2 = "HCA" if strpos(emp2,"CLEARLAKEREGIONALMEDICALCENTER")==1
replace emp2 = "ESSENTIAHEALTH" if strpos(emp2,"CLEARWATERVALLEYHOSPITAL")==1
replace emp2 = "THECLEVELANDCLINICHEALTHSYSTEM" if emp2=="CLEVELANDCLINCFLORIDA"
replace emp2 = "ROCHESTERREGIONALHEALTHSYSTEM" if strpos(emp2,"CLIFTONSPRINGSHOSPITAL")==1
replace emp2 = "LIFEPOINTHEALTH" if emp2=="CLINCHVALLEYMDCENTER"|emp2=="CLINCHVALLEYMEDICAL"
replace emp2 = "BASSETTHEALTHCARENETWORK" if strpos(emp2,"COBLESKILLREGIONALHOSPITAL")==1
replace emp2 = "HCA" if strpos(emp2,"COLISEUMMEDICAL")==1|emp2=="COLISEUMNORTHSIDE"
replace emp2 = "HCA" if strpos(emp2,"COLLETONMEDICAL")==1
replace emp2 = "SSMHEALTH" if strpos(emp2,"COLUMBUSCOMMUNITYHOSP")==1|emp2=="COLUMBUSCOMMHOSPITAL"
replace emp2 = "CONFLUENCEHEALTH" if strpos(emp2,"CONFLUENCEHEALTH")==1
replace emp2 = "HCA" if emp2=="CONROEREGIONALMEDICALCENTERCONROE"
replace emp2 = "COMMUNITYHOSPITALCORPORATION" if strpos(emp2,"CONTINUECAREHOSPITALAT")==1
replace emp2 = "COOKCHILDRENSHEALTHCARESYSTEM" if emp2=="COOKCHILDRENSHEALTHCARE"
replace emp2 = "ASCENSIONHEALTH" if strpos(emp2,"CRITTENTONHOSPITAL")==1
replace emp2 = "PROSPECTMEDICALHOLDINGS" if strpos(emp2,"CROZERCHESTERMED")==1
replace emp2 = "SUMMAHEALTHSYSTEM" if strpos(emp2,"CRYSTALCLINICORTHO")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="CYPRESSFAIRBANKSMEDICAL"
replace emp2 = "IASISHEALTHCARECORPORATION" if strpos(emp2,"DAVISHOSPITALANDMEDICAL")==1|strpos(emp2,"DAVISHOSPITALMEDICAL")==1
replace emp2 = "MERCYHEALTH" if strpos(emp2,"DEARBORNCOUNTYHOSPITAL")==1
replace emp2 = "HUNTSVILLEHOSPITALHEALTHSYSTEM" if emp2=="DECATURMORGANGENERALHOSPITAL"
replace emp2 = "DEKALBMEDICALCENTER" if strpos(emp2,"DEKALBMEDICA")==1
replace emp2 = "ADVENTISTHEALTH" if emp2=="DELANOREGIONALMEDICALC"
replace emp2 = "ASCENSIONHEALTH" if emp2=="DELLCHILDRENSMEDICALCENTER"
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="DELRAYMEDICAL"|emp2=="DESERTREGIONALMEDICALC"
replace emp2 = "BANNERHEALTH" if strpos(emp2,"DELTACOUNTYMEMORIALHOSPITAL")==1
replace emp2 = "INTERMOUNTAINHEALTHCARE" if strpos(emp2,"DIXIEREGIONALMEDICAL")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="DMCCHILDRENSHOSPITAL"|emp2=="DMCHURONVALLEYHEARTMILFORD"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="DOCTORSHOSPITALLAREDO"
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="DOCTORSHOSPITALMANTECA"
replace emp2 = "HCA" if emp2 =="DOCTORSHOSPITALSARASOTA"
replace emp2 = "DIGNITYHEALTH" if emp2=="DOMINICANHOSPITALSANTACRUZ"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="DOVERBEHAVIORALHEALTH"
replace emp2 = "HCA" if emp2=="EASTERNIDAHOREGIONALMEDICAL"|emp2=="EASTERNIDAHORMC"
replace emp2 = "EASTERNMAINEHEALTHCARESYSTEMS" if strpos(emp2,"EASTERNMAINEHEALTHCARE")==1
replace emp2 = "UNIVERSITYOFKENTUCKYHEALTHCARE" if strpos(emp2,"EASTERNSTATEHOSPITAL")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"EASTGEORGIAREGIONAL")==1
replace emp2 = "HCA" if strpos(emp2,"EASTHOUSTONMEDICALCENTER")==1|strpos(emp2,"EASTHOUSTONREGIONALMEDICALCENTER")==1
replace emp2 = "AVANTIHOSPITALS" if strpos(emp2,"EASTLOSANGELESDOCTORS")==1
replace emp2 = "HCA" if emp2=="EASTSIDEMEDICAL"
replace emp2 = "EASTTEXASMEDICALCENTERREGIONALHEALTHCARESYSTEM" if strpos(emp2,"EASTTEXASMEDICALCENTER")==1
replace emp2 = "RCCHHEALTHCAREPARTNERS" if strpos(emp2,"ELIZACOFFEEMEMORIAL")==1
replace emp2 = "ELMHURSTMEMORIALHEALTHCARE" if strpos(emp2,"ELMHURSTMEMORIALHEALTHCARE")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"EMERALDCOASTBEHAVIORAL")==1
replace emp2 = "HCA" if strpos(emp2,"ENGLEWOODCOMMUNITYHOSPITAL")==1
replace emp2 = "EPHRAIMMCDOWELLHEALTH" if strpos(emp2,"EPHRAIMMCDOWELL")==1
replace emp2 = "ESSENTIAHEALTH" if strpos(emp2,"ESSENTIAHEALTH")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="FAIRMOUNTBEHAVIORALHEALTHSY"
replace emp2 = "HCA" if strpos(emp2,"FAWCETTMEMORIALHOSPITAL")==1
replace emp2 = "ADVENTISTHEALTH" if emp2=="FEATHERRIVERHOSPITALADVENTISTHEALTH"
replace emp2 = "LIFEPOINTHEALTH" if emp2=="FLEMINGCOHOSPITAL"
replace emp2 = "ADVENTISTHEALTHSYSTEM" if emp2=="FLORIDAHOPSITALFLAGLER"
replace emp2 = "PROSPECTMEDICALHOLDINGS" if emp2=="FOOTHILLREGINALMEDICALCENRTER"|emp2=="FOOTHILLREGINALMEDICALCENTER"|emp2=="FOOTHILLREGIONALMEDICALCENRTER"|emp2=="FOOTHILLREGIONALMEDICALCENTERTUSTIN"|emp2=="FOOTHILLREGIONALHOSPITALTUSTIN"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="FORTLAUDERDALEHOSPITALAMERICA"
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="FOUNTAINVALLEYREGIONAL" | emp2=="FOUNTAINVALLEYREGIONALHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2== "FOXRUNCENTER"
replace emp2 = "HCA" if strpos(emp2,"FRANKFORTREGIONALMEDICAL")==1
replace emp2 = "ADVENTISTHEALTH" if emp2=="FRANKRHOWARDMEDICALMEMORIALHOSPITAL"
replace emp2 = "COMMUNITYMEDICALCENTERS" if emp2=="FRESNOHEART"|emp2=="FRESNOHEARTANDSURGICAL"
replace emp2 = "PRIMEHEALTHCARESERVICES" if strpos(emp2,"GARDENGROVEHOSPITAL")==1
replace emp2 = "CATHOLICHEALTHINITIATIVES" if strpos(emp2,"GARRISONMEMORIALHOSPITAL")==1
replace emp2 = "GENESISHEALTHSYSTEM" if strpos(emp2,"GENESISMEDICALCENTER")==1
replace emp2 = "FINGERLAKESHEALTH" if strpos(emp2,"FINGERSLAKESHEALTH")>0
replace emp2 = "ADVENTISTHEALTH" if emp2=="GLENDALEADVENTIST"|emp2=="GLENDALEADVENTISTMEDICA"
replace emp2 = "DIGNITYHEALTH" if emp2=="GLENDALEMEMHOSPANDHLTHCTR"|emp2=="GLENDALEMEMHOSPHLTHCTR"
replace emp2 = "GOODSHEPHERDHEALTHSYSTEM" if strpos(emp2,"GOODSHEPHERDMEDICAL")==1
replace emp2 = "TRINITYHEALTH" if strpos(emp2,"GOTTLIEBMEMORIALHOSPITA")==1
replace emp2 = "NEWYORKPRESBYTERIANHEALTHCARESYSTEM" if strpos(emp2,"GRACIESQUAREHOSP")==1
replace emp2 = "GRADYHEALTHSYSTEM" if emp2=="GRADYMEMORIAL"
replace emp2 = "UNITYPOINTHEALTH" if strpos(emp2,"GREATERREGIONALMEDICAL")==1
replace emp2 = "CATHOLICHEALTHINITIATIVES" if strpos(emp2,"GRIMESSTJOSEPHHEALTH")==1
replace emp2 = "GUNDERSENHEALTHSYSTEM" if strpos(emp2,"GUNDERSENBOSCO")==1|emp2=="GUNDERSENHEALTH"
replace emp2 = "GWINNETTHEALTHSYSTEM" if emp2=="GWINNETTMEDICALCENTER"
replace emp2 = "LOSANGELESCOUNTYHEALTHSERVICESDEPARTMENT" if strpos(emp2,"HARBORUCLA")==1
replace emp2 = "LIFEPOINTHEALTH" if strpos(emp2,"HARRISREGIONALHOSPITAL")==1
replace emp2 = "TRINITYHEALTH" if emp2=="HAWARDENREGIONALHEALTHC"
replace emp2 = "APPALACHIANREGIONALHEALTHCARE" if emp2=="HAZARDARHHOSPITAL"
replace emp2 = "AVERAHEALTH" if strpos(emp2,"HEGGMEMORIAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"HILLCRESTBEHAVIORAL")==1
replace emp2 = "HONORHEALTH" if strpos(emp2,"HONORHEALTH")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="HOUSTONNORTHWESTMEDICAL"
replace emp2 = "HOSPITALSISTERSHEALTHSYSTEM" if strpos(emp2,"HSHSS")==1
replace emp2 = "INSPIRAHEALTHNETWORK" if strpos(emp2,"INSPIRAMEDICAL")==1
replace emp2 = "INTEGRISHEALTH" if strpos(emp2,"INTEGRIS")==1 & (strpos(emp2,"HEALTH")>0 | strpos(emp2,"HOSPITAL")>0)
replace emp2 = "ASCENSIONHEALTH" if emp2=="JANEPHILLIPSMEDICALCENTER"
replace emp2 = "HCA" if emp2=="JOHNRANDOLPHMEDICALCENTERHOPEWELL"
replace emp2 = "JOHNSHOPKINSHEALTHSYSTEM" if strpos(emp2,"JOHNSHOPKINS")==1 & strpos(emp2,"HOSPITAL")>0
replace emp2 = "JOHNSHOPKINSHEALTHSYSTEM" if strpos(emp2,"JOHNSHOPKINS")==1 & strpos(emp2,"BAYVIEW")>0
replace emp2 = "HCA" if emp2=="JOHNSTONWILLISHOSPITALS"
replace emp2 = "ASCENSIONHEALTH" if emp2=="KANSASSURGERYANDRECOVERY"|emp2=="KANSASSURGERYRECOVERYCENTER"
replace emp2 = "RCCHHEALTHCAREPARTNERS" if emp2=="KERSHAWHEALTHMEDICALCENTER"
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="LAKEPOINTEMEDICALCENTE"
replace emp2 = "AVERAHEALTH" if strpos(emp2,"LANDMANNJUNGMANMEMORIAL")==1|strpos(emp2,"LANDMANNJUNGEMANMEMORIAL")==1
replace emp2 = "HCA" if strpos(emp2,"LARGOMEDICALCENTER")==1
replace emp2 = "HCA" if strpos(emp2,"LASPALMASDELSOLHEALTHCARE")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"LAURELOAKSBEHAVIORAL")==1
replace emp2 = "HCA" if strpos(emp2,"LAWNWOODREGIONALMED")==1
replace emp2 = "LEGACYHEALTH" if emp2 =="LEGACYHEALTHSYSTEM"
replace emp2 = "ADVENTISTHEALTH" if strpos(emp2,"LODIMEMORIAL")>0
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"LONGVIEWREGIONAL")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if strpos(emp2,"LOSALAMITOSMEDICAL")==1
replace emp2 = "LIFEPOINTHEALTH" if strpos(emp2,"LOSALAMOSMEDICAL")==1
replace emp2 = "HCA" if strpos(emp2,"LOSROBLESHOSPITALANDMEDICALCENTER")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"LOWERKEYSMEDICAL")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="LUTHERANHOSPITALINDIANA"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if strpos(emp2,"MADISONSTJOSEPH")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"MANATEEMEMORIAL")==1
replace emp2 = "UNITYPOINTHEALTH" if strpos(emp2,"MARENGOMEMORIALHOSPITAL")==1
replace emp2 = "LIFEPOINTHEALTH" if strpos(emp2,"MARIAPARHAMMEDICAL")==1
replace emp2 = "MEMORIALHEALTHSYSTEM" if strpos(emp2,"MARIETTAMEMORIAL")==1
replace emp2 = "CEDARS-SINAIHEALTHSYSTEM" if emp2=="MARINADELREYHOSPITAL"
replace emp2 = "APPALACHIANREGIONALHEALTHCARE" if emp2=="MARYBRECKINRIDGEHOSPITAL"
replace emp2 = "THECLEVELANDCLINICHEALTHSYSTEM" if strpos(emp2,"MARYMOUNTHOSPITAL")==1
replace emp2 = "PARTNERSHEALTHCARESYSTEM" if strpos(emp2,"MASSACHUSETTSGENERALHOSPITAL")==1
replace emp2 = "MAURYREGIONALMEDICALCENTER" if strpos(emp2,"MAURYREGIONALMEDICALCENTER")==1
replace emp2 = "MAYOCLINIC" if strpos(emp2,"MAYOCLINIC")==1
replace emp2 = "BAYLORSCOTTANDWHITEHEALTH" if emp2=="MCLANECHILDRENSHOSPITAL"
replace emp2 = "MCLARENHEALTHCARECORP" if strpos(emp2,"MCLARENBATYREGION")==1|strpos(emp2,"MCLARENFLINT")==1|strpos(emp2,"MCLARENGREATERLANSING")==1|emp2=="MCLARENHEALTHCAREAMERICA"|emp2=="MCLARENHEALTHCARE"|emp2=="MCLARENMACOMBMEDICALCENTER"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="MCNAIRYREGIONALMEDICALHOSPITAL"
replace emp2 = "ASCENSIONHEALTH" if emp2 =="MCPHERSONHOSITAL"
replace emp2 = "LIFEPOINTHEALTH" if emp2=="MEADOWVIEWREGIONALMEDICALCE"|emp2=="MEADOWVIEWREGIONALHOSPITAL"
replace emp2 = "MEDSTARHEALTH" if strpos(emp2,"MEDSTAR")==1 & strpos(emp2,"HOSPITAL")>0
replace emp2 = "MEMORIALHERMANNHEALTHCARESYSTEM" if strpos(emp2,"MEMORIALHERMANN")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"MICHIANABEHAVIORAL")==1
replace emp2 = "MIDMICHIGANHEALTH" if strpos(emp2,"MIDMICHIGANHEALTH")==1|strpos(emp2,"MIDMICHIGANMEDICAL")==1
replace emp2 = "SUTTERHEALTH" if emp2=="MILLSPENINSULAHEALTHSERVICESHEALTH"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if strpos(emp2,"MOBRIDGEREGIONALHOSPITA")==1
replace emp2 = "RWJBARNABASHEALTH" if strpos(emp2,"MONMOUTHMEDICALCENTER")==1
replace emp2 = "PRIMEHEALTHCARESERVICES" if strpos(emp2,"MONTCLAIRHOSPITALMEDICALCEN")==1
replace emp2 = "TRINITYHEALTH" if strpos(emp2,"MORTONPLANTHOSP")==1
replace emp2 = "CAROLINASHEALTHCARESYSTEM" if strpos(emp2,"MOSESCONE")==1
replace emp2 = "IASISHEALTHCARECORPORATION" if strpos(emp2,"MOUNTAINVISTAMEDICAL")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="MUNROEREGIONALMEDICALC"
replace emp2 = "NEWYORKPRESBYTERIANHEALTHCARESYSTEM" if strpos(emp2,"NEWYORKPRESBYTERIAN")==1
replace emp2 = "PROSPECTMEDICALHOLDINGS" if strpos(emp2,"NIXHEALTH")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if strpos(emp2,"NORTHCENTRALSURGICALCENTER")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="NORTHEASTREGIONALMEDICALCEN"
replace emp2 = "HCA" if strpos(emp2,"NORTHHILLSHOSPITAL")==1
replace emp2 = "DIGNITYHEALTH" if strpos(emp2,"NORTHRIDGEHOSPITAL")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if strpos(emp2,"NORTHSHOREMEDICALCENTE")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"NORTHSPRINGBEHAVIORAL")==1|strpos(emp2,"NORTHSTARBEHAVIORAL")==1
replace emp2 = "SSMHEALTH" if strpos(emp2,"NORTHTEXASMEDICAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="NORTHWESTTEXASHEALTHCARE"
replace emp2 = "HCA" if strpos(emp2,"OAKHILLHOSPITAL")==1
replace emp2 = "DIGNITYHEALTH" if strpos(emp2,"OAKVALLEYHOSPITALDISTR")==1
replace emp2 = "MERCYHEALTH" if strpos(emp2,"OKLAHOMAHEARTHOSP")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="OLDVINEYARDBEHAVIORALHOSPITAL"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"COMMUNITYHEALTHSYSTEM")>0|emp2=="OROVALLEYHOSPITAL"
replace emp2="CATHOLICHEALTHINITIATIVES" if emp2=="ORTHOCOLORADO"
replace emp2 = "OSFHEALTHCARESYSTEM" if strpos(emp2,"OSFHEALTHCARE")==1|strpos(emp2,"OSFSAINT")==1|strpos(emp2,"OSFST")==1
replace emp2 = "HCA" if strpos(emp2,"OVERLANDPARKREGIONALM")==1
replace emp2 = "SSMHEALTH" if strpos(emp2,"OZARKSMEDICAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"PALMDALEREGIONAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"PALMETTOLOWCOUNTRYBEHAVIORAL")==1
replace emp2 = "HCA" if strpos(emp2,"PALMSOFPASADENAHOSPITAL")==1|emp2=="PALMSPASADENAHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"PALOVERDEBEHAVIORAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"PARKWOODBEHAVIORAL")==1
replace emp2 = "LIFEPOINTHEALTH" if strpos(emp2,"PERSONMEMORIAL")==1
replace emp2 = "NORTHWELLHEALTH" if emp2=="PHELPSMEMORIALHEALTHCENTER"
replace emp2 = "SSMHEALTH" if emp2=="PIKECOUNTYMEMORIALHOSPITA"
replace emp2 = "IASISHEALTHCARECORPORATION" if strpos(emp2,"PIKESPEAKREGIONALHOSPITAL")==1
replace emp2 = "AVERAHEALTH" if strpos(emp2,"PIPESTONECOUNTYMEDICALCENTER")==1
replace emp2 = "HCA" if strpos(emp2,"PLANTATIONGENERALHOSPITAL")==1
replace emp2  = "HCA" if strpos(emp2,"PRESBYTERIAN")==1 & strpos(emp2,"LUKE")>0
replace emp2 = "PROVIDENCEHEALTHANDSERVICES" if emp2=="PROVIDENCEALASKA"
replace emp2 = "HCA" if emp2=="PUTNAMCOMMUNITYMEDICAL"|emp2=="RAULERSONHOSPITALOKEECHOBEE"|emp2=="REDMONDREGIONALMEDICALCENTERROME"
replace emp2 = "TRINITYHEALTH" if emp2=="REGIONALHEALTHSVCSOFHOWARDCO"
replace emp2 = "REGIONALHOSPITALJACKSON" if strpos(emp2,"REGIONALHOSPITAL")==1 & strpos(emp2,"JACKSON")>0
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="REGIONALHOSPITALOFSCRANTONUSA"
replace emp2 = "HCA" if emp2=="REGIONALMEDICALCENTERATBAYONETPOINT"|strpos(emp2,"REGIONALMEDICALCENTERBAYONET")==1|strpos(emp2,"REGIONALMEDICALCENTEROFSANJOSE")==1|strpos(emp2,"REGIONALMEDICALCENTERSANJOSE")==1|emp2=="RESTONHOSPITALCENTERRESTON"|emp2=="RIVERSIDECOMMUNITYHOSPITALHCA"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"RIVENDELLBEHAVIORAL")==1|strpos(emp2,"RIVEROAKSHOSPITAL")==1|strpos(emp2,"RIVERPARKHOSPITAL")==1
replace emp2 = "CAROLINASHEALTHCARESYSTEM" if strpos(emp2,"ROPERSTFRANCIS")==1
replace emp2 = "HCA" if strpos(emp2,"ROSEMEDICALCENTER")==1
replace emp2 = "PRIMEHEALTHCARESERVICES" if emp2=="ROXBOROUGHMEMORIALHOSPITALPHILADELPHIAPA"
replace emp2 = "LIFEPOINTHEALTH" if emp2=="RUTHERFORDREGHEALTHSYS"|emp2=="RUTHERFORDREGIONAL"|emp2=="RUTHERFORDREGIONALMEDICALCENTER"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="SAINTJOSEPHLONDONSAINTJOSEPHMARTIN"
replace emp2 = "RCCHHEALTHCAREPARTNERS" if emp2=="SALINEMEMORIAL"|emp2=="SALINEMEMORIALHOSPITAL"
replace emp2 = "PRIMEHEALTHCARESERVICES" if emp2=="SANDIMASCOMMUNITYHOSTPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="SANDYPINESHOSPITAL"
replace emp2 = "SANFORDHEALTH" if emp2=="SANFORDABERDEEN"
replace emp2 = "SANFORDHEALTH" if strpos(emp2,"SANFORD")==1 & strpos(emp2,"MEDICALCENTER")>0
replace emp2 = "BAYLORSCOTTANDWHITEHEALTH" if emp2=="SCOTTANDWHITEHEALTHCARE"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="SEBASTIANRIVERMEDICAL"
replace emp2 = "ASCENSIONHEALTH" if emp2=="SETONHIGHLANDLAKESHOSPITAL"|emp2=="SETONMEDICALCTRAUSTIN"|emp2=="SETONMEDICALCTRHAYS"|emp2=="SETONMEDICALCTRWILLIAMSON"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"SHADOWMOUNTAINBEHAVIORALHEALTH")==1
replace emp2 ="RCCHHEALTHCAREPARTNERS" if strpos(emp2,"SHARONHOSPITAL")==1
replace emp2 = "ADVENTISTHEALTH" if emp2=="SIMIVALLEYHOSPITALADVENTIST"
replace emp2 = "HCA" if strpos(emp2,"SKYRIDGEMEDICAL")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="SOUTHBALDWINREGIONALMEDICAL"
replace emp2 = "PROSPECTMEDICALHOLDINGS" if strpos(emp2,"SOUTHERNCALIFORNIAHOSPITAL")==1|strpos(emp2,"SOUTHERNCALIFORNIAHHOSPITAL")==1
replace emp2 = "HCA" if strpos(emp2,"SOUTHERNHILLSHOSP")==1
replace emp2 = "PRIMEHEALTHCARESERVICES" if strpos(emp2,"SOUTHERNREGIONALMEDICAL")==1
replace emp2 = "LIFEPOINTHEALTH" if strpos(emp2,"SOUTHERNTENNESSEEREGIONALHEALTH")==1
replace emp2 = "NORTHWELLHEALTH" if strpos(emp2,"SOUTHOAKSHOSPITAL")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"SOUTHTEXASHEALTHSYSTEM")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="SPARKSMEDICALCENTER"
replace emp2 = "HCA" if strpos(emp2,"SPOTSYLVANIAREGIONALMEDICAL")==1
replace emp2 = "SSMHEALTH" if emp2=="SSMCARDINALGLENNONCHILDRENSHOSPITAL"|emp2=="SSMDEPAUL"|strpos(emp2,"SSMHEALTHCARDINALGLENNON")==1|emp2=="SSMHEALTHCARE"|emp2=="SSMHEALTHCARESYSTEM"|strpos(emp2,"SSMHEALTH")==1|emp2=="SSMOKLAHOMASAINTANTHONYSHOSPITAL"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="STANTHONYNORTHHOSPITAL"
replace emp2 = "TENETHEALTHCARECORPORATION" if strpos(emp2,"STCHRISTOPHERSHOSPITALFORCHILDREN")==1
replace emp2 = "UNITYPOINTHEALTH" if emp2=="STEWARTMEMORIALCOMMUNITYHOSPITALANDMCCRARYROSTCLINICS"
replace emp2 = "ADVENTISTHEALTH" if emp2=="STHELENAHOSPITALBEHAVIORALHEALTH"
replace emp2 = "ASCENSIONHEALTH" if emp2=="STJOHNHOSPITALANDMEDCEN"
replace emp2 = "TRINITYHEALTH" if emp2=="STMARYSGOODSAMARITAN"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"STREAMWOODBEHAVIORAL")==1
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="STRINGFELLOWMEMORALHOSPITAL"
replace emp2 = "MERCYHEALTH" if emp2=="STRITASMEDICALCENTE"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="SUMMERLINHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="SUNCOASTBEHAVIORALHEALTH"
replace emp2 = "SUTTERHEALTH" if strpos(emp2,"SUTTERLAKESIDE")==1
replace emp2 = "HCA" if emp2=="SWEDISHMEDICAL"
replace emp2 = "CATHOLICHEALTHINITIATIVES" if emp2=="TAYLORREGIONAL"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if strpos(emp2,"TENNOVAHEALTHCARE")==1
replace emp2 = "TEXASHEALTHRESOURCES" if strpos(emp2,"TEXASHEALTH")==1
replace emp2 = "HCA" if emp2=="TEXASHOUSTONTEXASORTHOPEDICHOSPITAL"
replace emp2 = "UNIVERSALHEALTHSERVICES" if emp2=="THEBROOKHOSPITALSDUPONT"
replace emp2 = "THEHOSPITALSOFPROVIDENCE" if emp2=="THEHOSPITALSOFPROVIDEN"
replace emp2 = "HCA" if strpos(emp2,"THOUSANDOAKSSURGICAL")==1|emp2=="THOUSANDOAKSTHOUSANDSURGICALHOSPITALACAMPUSOFLOSROBLESHOSPITALANDMEDICALCENTER"|emp2=="TIMPANOGOSREGIONALMEDICALCTR"
replace emp2 = "COMMUNITYHEALTHSYSTEMS" if emp2=="TURKEYCREEKMEDICALCENTERTENNOVA"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"UHS")==1 & (strpos(emp2,"HOSPITAL")>0 |strpos(emp2,"HEALTH")>0)
replace emp2 = "UNITYPOINTHEALTH" if strpos(emp2,"UNITYPOINTHEALTH")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="VALLEYBAPTISTMEDICALCENTER"
replace emp2 ="SSMHEALTH" if emp2=="WASHINGTONCOUNTYHOSPITA"
replace emp2 = "HCA" if emp2=="WESLEYWOODLAWNHOSPITAL26ER"|strpos(emp2,"WESTHILLSHOSPITALHCA")==1|strpos(emp2,"WESTHILLSHOSPITALANDMEDICALCENTERHCA")==1
replace emp2 = "TENETHEALTHCARECORPORATION" if emp2=="WESTSUBURBANMEDICALCEN"
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"WESTWOODLODGEHOSPITAL")>0
replace emp2 = "ASCENSIONHEALTH" if strpos(emp2,"WHEATONFRANCISCANHEALTHCARE")==1
replace emp2 = "UNIVERSALHEALTHSERVICES" if strpos(emp2,"WINDSORLAURELWOODCENTER")==1|strpos(emp2,"WINDSORLAURELWOODHOSP")==1
replace emp2 ="MAYOCLINIC" if emp2=="YAMPAVALLEYMEDICAL"

replace emp2 = subinstr(emp2,"HEALTHSYSTEMS","HEALTHSYSTEM",.) if substr(emp2,-13,13)=="HEALTHSYSTEMS"

********************************************************************************

*Set emp to the corrected emp2 for the hospital names
*need to be careful with this line because it can overwrite our earlier cleaning!
replace emp = emp2 if employer != emp2

summarize N

********************************************************************************

*set everything to lowercase
replace emp = strlower(emp)

*take out all characters except for 0-9 and a-z, only need to do 130 because anything longer than 128 was truncated
forvalues i = 1/130 {
	replace emp = regexr(emp, "[^0-9a-z]", "")
}

replace emp = stritrim(emp)
replace emp = strtrim(emp)

********************************************************************************

* Emp to employer-original crosswalk
* This contains counts of numbers of occurences, and flags for vacancy data
keep emp employer_original N

summarize N

* Counts of emp (not employer_original)
bys emp: egen sum_N = total(N)
drop N
rename sum_N N

summarize N

di "number of unique employer originals"
distinct employer_original

*cleanup
duplicates drop

di "checks for problematic firm names which aren't being cleaned right"
tab employer_original if ustrregexm(employer_original, "'CPR and First Aid'")
tab employer_original if ustrregexm(employer_original, "'Jennifer Aniston' for")

save "clean_firm_names.dta", replace
