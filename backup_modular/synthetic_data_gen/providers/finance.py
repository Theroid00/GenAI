"""
Finance data provider for synthetic data generation.
This module contains data and functions to generate realistic financial data
such as transaction information, bank accounts, stock data, and other finance-specific information.
"""

import random
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import numpy as np

# ===== Financial Institutions =====

# Major global banks
GLOBAL_BANKS = [
    "JPMorgan Chase", "Bank of America", "Citigroup", "Wells Fargo", "Goldman Sachs",
    "Morgan Stanley", "HSBC", "Barclays", "Deutsche Bank", "Credit Suisse",
    "UBS", "BNP Paribas", "Société Générale", "Santander", "ING Group",
    "BBVA", "Standard Chartered", "Mitsubishi UFJ Financial Group", "Mizuho Financial Group",
    "Sumitomo Mitsui Financial Group", "Royal Bank of Canada", "Toronto-Dominion Bank",
    "Bank of China", "Industrial and Commercial Bank of China", "China Construction Bank"
]

# US Banks
US_BANKS = [
    "JPMorgan Chase", "Bank of America", "Citigroup", "Wells Fargo", "Goldman Sachs",
    "Morgan Stanley", "U.S. Bancorp", "PNC Financial Services", "Capital One", "TD Bank",
    "Bank of New York Mellon", "Charles Schwab", "BB&T", "SunTrust Banks", "State Street",
    "American Express", "Ally Financial", "Citizens Financial Group", "Fifth Third Bank",
    "KeyCorp", "Regions Financial", "Northern Trust", "M&T Bank", "Huntington Bancshares",
    "Synchrony Financial", "Discover Financial", "Comerica", "Zions Bancorporation"
]

# European Banks
EUROPEAN_BANKS = [
    "HSBC", "Barclays", "Deutsche Bank", "Credit Suisse", "UBS", "BNP Paribas",
    "Société Générale", "Santander", "ING Group", "BBVA", "Standard Chartered",
    "Commerzbank", "Crédit Agricole", "UniCredit", "Intesa Sanpaolo", "Nordea",
    "Lloyds Banking Group", "RBS Group", "KBC Group", "Danske Bank", "DNB ASA",
    "Banco Sabadell", "CaixaBank", "Bankinter", "ABN AMRO", "Rabobank"
]

# Asian Banks
ASIAN_BANKS = [
    "Mitsubishi UFJ Financial Group", "Mizuho Financial Group", "Sumitomo Mitsui Financial Group",
    "Bank of China", "Industrial and Commercial Bank of China", "China Construction Bank",
    "Agricultural Bank of China", "Bank of Communications", "DBS Group", "OCBC Bank",
    "United Overseas Bank", "State Bank of India", "ICICI Bank", "HDFC Bank", "Axis Bank",
    "Shinhan Bank", "KB Kookmin Bank", "Hana Financial Group", "Woori Bank", "Maybank",
    "CIMB Group", "Public Bank Berhad", "Siam Commercial Bank", "Bangkok Bank", "Kasikornbank"
]

# ===== Payment Methods =====

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "Bank Transfer", "Cash", "Check", "Money Order",
    "PayPal", "Venmo", "Zelle", "Apple Pay", "Google Pay", "Samsung Pay",
    "Bitcoin", "Ethereum", "Cryptocurrency", "Wire Transfer", "ACH Transfer",
    "Gift Card", "Store Credit", "Payment Plan"
]

# ===== Transaction Types =====

TRANSACTION_TYPES = [
    "Purchase", "Refund", "Payment", "Withdrawal", "Deposit", "Transfer",
    "Subscription", "Recurring Payment", "Bill Payment", "Payroll", "Interest",
    "Dividend", "Fee", "Tax", "Investment", "Loan", "Insurance", "Rent",
    "Mortgage", "Utility"
]

# ===== Merchant Categories =====

MERCHANT_CATEGORIES = [
    "Grocery", "Restaurant", "Retail", "Gas Station", "Utilities", "Telecom",
    "Healthcare", "Entertainment", "Travel", "Transportation", "Lodging",
    "Education", "Professional Services", "Home Improvement", "Insurance",
    "Financial Services", "Charity", "Government", "Technology", "Subscription Services"
]

# ===== Stock Market Data =====

# Major stock exchanges
STOCK_EXCHANGES = [
    "NYSE", "NASDAQ", "LSE", "TSE", "SSE", "HKEX", "Euronext", "BSE", "NSE", "FWB"
]

# Sample stock symbols
STOCK_SYMBOLS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "INTC",
    # Financial
    "JPM", "BAC", "C", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "TMO", "UNH", "CVS", "ABT", "MDT", "AMGN",
    # Consumer
    "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "NFLX", "HD", "WMT"
]

# ===== Currency Data =====

# Major currencies with codes
CURRENCIES = [
    ("USD", "US Dollar", "$"),
    ("EUR", "Euro", "€"),
    ("JPY", "Japanese Yen", "¥"),
    ("GBP", "British Pound", "£"),
    ("AUD", "Australian Dollar", "A$"),
    ("CAD", "Canadian Dollar", "C$"),
    ("CHF", "Swiss Franc", "Fr"),
    ("CNY", "Chinese Yuan", "¥"),
    ("HKD", "Hong Kong Dollar", "HK$"),
    ("NZD", "New Zealand Dollar", "NZ$")
]

# ===== Financial Class Generator =====

class FinanceDataProvider:
    """Provider for finance-specific data generation."""
    
    @staticmethod
    def generate_bank(region: Optional[str] = None) -> str:
        """
        Generate a random bank name.
        
        Args:
            region: Optional region ('us', 'europe', 'asia', or None for global)
            
        Returns:
            A random bank name
        """
        if region:
            region = region.lower()
            if region == 'us':
                return random.choice(US_BANKS)
            elif region in ['europe', 'eu']:
                return random.choice(EUROPEAN_BANKS)
            elif region == 'asia':
                return random.choice(ASIAN_BANKS)
        
        return random.choice(GLOBAL_BANKS)
    
    @staticmethod
    def generate_account_number() -> str:
        """Generate a random bank account number."""
        # Format: 10-12 digits
        length = random.randint(10, 12)
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    
    @staticmethod
    def generate_routing_number() -> str:
        """Generate a random bank routing number (ABA)."""
        # Format: 9 digits
        return ''.join([str(random.randint(0, 9)) for _ in range(9)])
    
    @staticmethod
    def generate_iban(country: Optional[str] = None) -> str:
        """
        Generate a random IBAN (International Bank Account Number).
        
        Args:
            country: Optional country code (e.g., 'DE', 'FR', 'GB')
            
        Returns:
            A random IBAN
        """
        if not country:
            country = random.choice(['DE', 'FR', 'GB', 'ES', 'IT', 'NL'])
        
        # Format: Country code + 2 check digits + bank code + account number
        country_code = country
        check_digits = f"{random.randint(0, 99):02d}"
        
        if country == 'GB':  # UK
            bank_code = ''.join([str(random.randint(0, 9)) for _ in range(4)])
            sort_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
            return f"{country_code}{check_digits}{bank_code}{sort_code}{account_number}"
        elif country == 'DE':  # Germany
            bank_code = ''.join([str(random.randint(0, 9)) for _ in range(8)])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
            return f"{country_code}{check_digits}{bank_code}{account_number}"
        elif country == 'FR':  # France
            bank_code = ''.join([str(random.randint(0, 9)) for _ in range(5)])
            branch_code = ''.join([str(random.randint(0, 9)) for _ in range(5)])
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(11)])
            check = ''.join([str(random.randint(0, 9)) for _ in range(2)])
            return f"{country_code}{check_digits}{bank_code}{branch_code}{account_number}{check}"
        else:
            # Generic format for other countries
            rest_digits = random.randint(10, 25)  # Variable length for different countries
            rest = ''.join([str(random.randint(0, 9)) for _ in range(rest_digits)])
            return f"{country_code}{check_digits}{rest}"
    
    @staticmethod
    def generate_swift_bic() -> str:
        """Generate a random SWIFT/BIC code."""
        # Format: 8 or 11 characters
        # First 4: bank code (letters)
        # Next 2: country code (letters)
        # Next 2: location code (letters or digits)
        # Last 3: branch code (optional, letters or digits)
        
        bank_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
        country_code = random.choice(['US', 'GB', 'DE', 'FR', 'JP', 'CN', 'CA', 'CH', 'AU', 'IT'])
        location_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=2))
        
        # 50% chance to include branch code
        if random.random() < 0.5:
            branch_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=3))
            return f"{bank_code}{country_code}{location_code}{branch_code}"
        else:
            return f"{bank_code}{country_code}{location_code}"
    
    @staticmethod
    def generate_credit_card(card_type: Optional[str] = None) -> Dict[str, str]:
        """
        Generate random credit card information.
        
        Args:
            card_type: Optional card type ('visa', 'mastercard', 'amex', 'discover')
            
        Returns:
            Dictionary with credit card details
        """
        # Note: This generates fake credit card formats that won't pass validation
        if not card_type:
            card_types = ["Visa", "MasterCard", "American Express", "Discover"]
            card_type = random.choice(card_types)
        else:
            card_type = card_type.lower()
            if card_type == 'visa':
                card_type = "Visa"
            elif card_type in ['mastercard', 'master']:
                card_type = "MasterCard"
            elif card_type in ['amex', 'american express']:
                card_type = "American Express"
            elif card_type == 'discover':
                card_type = "Discover"
            else:
                card_type = "Visa"  # Default to Visa
        
        # Generate number pattern based on card type
        if card_type == "American Express":
            # AmEx starts with 34 or 37 and has 15 digits
            prefix = random.choice(["34", "37"])
            digits = ''.join(random.choices('0123456789', k=13))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:10]} {number[10:]}"
        elif card_type == "Visa":
            # Visa starts with 4 and has 16 digits
            prefix = "4"
            digits = ''.join(random.choices('0123456789', k=15))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        elif card_type == "MasterCard":
            # MasterCard starts with 51-55 and has 16 digits
            prefix = f"5{random.randint(1, 5)}"
            digits = ''.join(random.choices('0123456789', k=14))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        else:  # Discover
            # Discover starts with 6011 and has 16 digits
            prefix = "6011"
            digits = ''.join(random.choices('0123456789', k=12))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        
        # Generate expiration date
        exp_month = random.randint(1, 12)
        exp_year = random.randint(2023, 2030)
        
        # Generate CVV
        cvv = ''.join(random.choices('0123456789', k=3 if card_type != "American Express" else 4))
        
        return {
            'type': card_type,
            'number': number,
            'formatted': formatted,
            'expiration': f"{exp_month:02d}/{exp_year}",
            'cvv': cvv
        }
    
    @staticmethod
    def generate_transaction() -> Dict[str, Any]:
        """Generate a random financial transaction."""
        # Generate amount
        amount = round(random.uniform(1.0, 1000.0), 2)
        
        # Generate transaction type
        transaction_type = random.choice(TRANSACTION_TYPES)
        
        # Generate merchant information
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        merchant_name = f"{random.choice(['Shop', 'Store', 'Market', 'Services', 'Company'])} {random.randint(1, 999)}"
        
        # Generate payment method
        payment_method = random.choice(PAYMENT_METHODS)
        
        # Generate date within the past year
        days_ago = random.randint(0, 365)
        transaction_date = datetime.now() - timedelta(days=days_ago)
        date_str = transaction_date.strftime("%Y-%m-%d")
        time_str = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        
        # Generate status
        status = random.choice(["Completed", "Pending", "Failed", "Declined", "Refunded"])
        
        # Generate currency
        currency_info = random.choice(CURRENCIES)
        currency_code = currency_info[0]
        currency_symbol = currency_info[2]
        
        return {
            'amount': amount,
            'formatted_amount': f"{currency_symbol}{amount:.2f}",
            'currency': currency_code,
            'type': transaction_type,
            'merchant': merchant_name,
            'merchant_category': merchant_category,
            'payment_method': payment_method,
            'date': date_str,
            'time': time_str,
            'status': status,
            'reference': f"TXN{random.randint(10000000, 99999999)}"
        }
    
    @staticmethod
    def generate_stock_data() -> Dict[str, Any]:
        """Generate random stock market data."""
        # Generate stock symbol
        symbol = random.choice(STOCK_SYMBOLS)
        
        # Generate exchange
        exchange = random.choice(STOCK_EXCHANGES)
        
        # Generate price
        price = round(random.uniform(10.0, 1000.0), 2)
        
        # Generate price change
        change_percent = round(random.uniform(-5.0, 5.0), 2)
        change_amount = round(price * change_percent / 100, 2)
        
        # Generate volume
        volume = random.randint(100000, 10000000)
        
        # Generate market cap
        market_cap = price * volume * random.uniform(0.1, 10)
        market_cap_formatted = f"${market_cap/1000000000:.2f}B" if market_cap >= 1000000000 else f"${market_cap/1000000:.2f}M"
        
        # Generate date
        trading_date = datetime.now() - timedelta(days=random.randint(0, 30))
        date_str = trading_date.strftime("%Y-%m-%d")
        
        return {
            'symbol': symbol,
            'company_name': f"{symbol} Inc.",
            'exchange': exchange,
            'price': price,
            'formatted_price': f"${price:.2f}",
            'change_amount': change_amount,
            'change_percent': change_percent,
            'formatted_change': f"{'+' if change_amount >= 0 else ''}{change_amount:.2f} ({'+' if change_percent >= 0 else ''}{change_percent:.2f}%)",
            'volume': volume,
            'formatted_volume': f"{volume:,}",
            'market_cap': market_cap,
            'formatted_market_cap': market_cap_formatted,
            'date': date_str
        }
    
    @staticmethod
    def generate_currency_exchange() -> Dict[str, Any]:
        """Generate random currency exchange rate data."""
        # Select base and target currencies
        all_currencies = CURRENCIES.copy()
        base_currency = random.choice(all_currencies)
        all_currencies.remove(base_currency)
        target_currency = random.choice(all_currencies)
        
        # Generate exchange rate
        rate = round(random.uniform(0.01, 100.0), 4)
        
        # Generate date
        exchange_date = datetime.now() - timedelta(days=random.randint(0, 30))
        date_str = exchange_date.strftime("%Y-%m-%d")
        
        return {
            'base_currency_code': base_currency[0],
            'base_currency_name': base_currency[1],
            'target_currency_code': target_currency[0],
            'target_currency_name': target_currency[1],
            'rate': rate,
            'formatted_rate': f"1 {base_currency[0]} = {rate:.4f} {target_currency[0]}",
            'inverse_rate': round(1 / rate, 4),
            'date': date_str
        }
    
    @staticmethod
    def generate_loan_data() -> Dict[str, Any]:
        """Generate random loan information."""
        # Loan types
        loan_types = ["Mortgage", "Auto Loan", "Personal Loan", "Student Loan", "Business Loan", "Home Equity Loan"]
        loan_type = random.choice(loan_types)
        
        # Loan amount
        if loan_type == "Mortgage":
            amount = round(random.uniform(100000, 1000000), 2)
        elif loan_type == "Auto Loan":
            amount = round(random.uniform(10000, 50000), 2)
        elif loan_type == "Personal Loan":
            amount = round(random.uniform(1000, 30000), 2)
        elif loan_type == "Student Loan":
            amount = round(random.uniform(5000, 100000), 2)
        elif loan_type == "Business Loan":
            amount = round(random.uniform(10000, 500000), 2)
        else:  # Home Equity
            amount = round(random.uniform(20000, 200000), 2)
        
        # Interest rate
        interest_rate = round(random.uniform(2.0, 12.0), 2)
        
        # Term in months
        if loan_type == "Mortgage":
            term = random.choice([180, 240, 360])  # 15, 20, or 30 years
        elif loan_type == "Auto Loan":
            term = random.choice([36, 48, 60, 72])  # 3-6 years
        elif loan_type == "Personal Loan":
            term = random.choice([12, 24, 36, 48, 60])  # 1-5 years
        elif loan_type == "Student Loan":
            term = random.choice([120, 180, 240])  # 10-20 years
        elif loan_type == "Business Loan":
            term = random.choice([12, 24, 36, 48, 60, 84, 120])  # 1-10 years
        else:  # Home Equity
            term = random.choice([60, 120, 180, 240])  # 5-20 years
        
        # Calculate monthly payment (simplified)
        monthly_interest_rate = interest_rate / 100 / 12
        monthly_payment = amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** term) / ((1 + monthly_interest_rate) ** term - 1)
        monthly_payment = round(monthly_payment, 2)
        
        # Generate start date
        start_date = datetime.now() - timedelta(days=random.randint(0, 365 * 5))  # Within the past 5 years
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Calculate end date
        end_date = start_date + timedelta(days=30 * term)
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        return {
            'type': loan_type,
            'amount': amount,
            'formatted_amount': f"${amount:,.2f}",
            'interest_rate': interest_rate,
            'term_months': term,
            'term_years': round(term / 12, 1),
            'monthly_payment': monthly_payment,
            'formatted_monthly_payment': f"${monthly_payment:,.2f}",
            'start_date': start_date_str,
            'end_date': end_date_str,
            'status': random.choice(["Active", "Paid Off", "Defaulted", "Refinanced"])
        }
    
    @staticmethod
    def generate_investment_portfolio() -> Dict[str, Any]:
        """Generate a random investment portfolio."""
        # Number of holdings
        num_holdings = random.randint(3, 10)
        
        # Generate holdings
        holdings = []
        total_value = 0
        
        for _ in range(num_holdings):
            stock_data = FinanceDataProvider.generate_stock_data()
            shares = random.randint(1, 1000)
            position_value = stock_data['price'] * shares
            total_value += position_value
            
            holdings.append({
                'symbol': stock_data['symbol'],
                'shares': shares,
                'price': stock_data['price'],
                'value': position_value,
                'formatted_value': f"${position_value:,.2f}"
            })
        
        # Calculate percentages
        for holding in holdings:
            holding['percentage'] = round((holding['value'] / total_value) * 100, 2)
        
        # Sort by value (descending)
        holdings.sort(key=lambda x: x['value'], reverse=True)
        
        # Generate portfolio performance
        ytd_return = round(random.uniform(-15.0, 30.0), 2)
        one_year_return = round(random.uniform(-20.0, 40.0), 2)
        three_year_return = round(random.uniform(-10.0, 60.0), 2)
        five_year_return = round(random.uniform(0.0, 100.0), 2)
        
        return {
            'total_value': total_value,
            'formatted_total_value': f"${total_value:,.2f}",
            'holdings': holdings,
            'num_holdings': num_holdings,
            'performance': {
                'ytd_return': ytd_return,
                'one_year_return': one_year_return,
                'three_year_return': three_year_return,
                'five_year_return': five_year_return
            }
        }

# ===== Helper Functions =====

def get_generator(data_type: str) -> Optional[Callable]:
    """
    Get a generator function for specific finance data types.
    
    Args:
        data_type: Type of data to generate
        
    Returns:
        A generator function for the specified data type
    """
    # Map data types to provider methods
    provider = FinanceDataProvider()
    
    generators = {
        'bank': provider.generate_bank,
        'account_number': provider.generate_account_number,
        'routing_number': provider.generate_routing_number,
        'iban': provider.generate_iban,
        'swift': provider.generate_swift_bic,
        'swift_bic': provider.generate_swift_bic,
        'credit_card': provider.generate_credit_card,
        'transaction': provider.generate_transaction,
        'stock': provider.generate_stock_data,
        'stock_data': provider.generate_stock_data,
        'currency_exchange': provider.generate_currency_exchange,
        'exchange_rate': provider.generate_currency_exchange,
        'loan': provider.generate_loan_data,
        'loan_data': provider.generate_loan_data,
        'investment': provider.generate_investment_portfolio,
        'portfolio': provider.generate_investment_portfolio,
        'investment_portfolio': provider.generate_investment_portfolio
    }
    
    return generators.get(data_type.lower())

def get_supported_data_types() -> List[str]:
    """
    Get a list of all supported finance data types.
    
    Returns:
        A list of supported data types
    """
    return list(get_generator('').__globals__['generators'].keys())
