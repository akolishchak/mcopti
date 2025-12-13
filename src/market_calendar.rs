use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Weekday};
use chrono_tz::{America::New_York, Tz};

pub struct USMarketCalendar {
    start_year: i32,
    end_year: i32,
    holidays: Vec<Vec<NaiveDate>>,
    yearly_closes: Vec<Vec<NaiveDate>>,
}

impl USMarketCalendar {
    pub fn new(start_year: i32, end_year: i32) -> Self {
        let capacity = (end_year - start_year + 1) as usize;
        let mut holidays = Vec::with_capacity(capacity);
        let mut yearly_closes = Vec::with_capacity(capacity);
        for year in start_year..=end_year {
            let thanksgiving = Self::nth_weekday(year, 11, 3, 4);
            holidays.push(vec![
                Self::observed(NaiveDate::from_ymd_opt(year, 1, 1).unwrap()), // New Year
                Self::nth_weekday(year, 1, 0, 3), // MLK (3rd Mon Jan)
                Self::nth_weekday(year, 2, 0, 3), // Presidents (3rd Mon Feb)
                Self::easter(year) - Duration::days(2), // Good Friday
                Self::last_weekday(year, 5, 0), // Memorial (last Mon May)
                Self::observed(NaiveDate::from_ymd_opt(year, 6, 19).unwrap()), // Juneteenth
                Self::observed(NaiveDate::from_ymd_opt(year, 7, 4).unwrap()), // Independence
                Self::nth_weekday(year, 9, 0, 1), // Labor (1st Mon Sep)
                thanksgiving, // Thanksgiving (Thu)
                Self::observed(NaiveDate::from_ymd_opt(year, 12, 25).unwrap()), // Christmas
            ]);
            yearly_closes.push(vec![
                thanksgiving + Duration::days(1), // Black Friday
                NaiveDate::from_ymd_opt(year, 12, 24).unwrap(), // Christmas Eve
            ]); 
        }
        Self {
            start_year,
            end_year,
            holidays,
            yearly_closes,
        }
    }

    pub fn is_trading_day(&self, date: NaiveDate) -> bool {
        let year = date.year();
        assert!(year >= self.start_year && year <= self.end_year);

        date.weekday().num_days_from_monday() < 5
            && !self.holidays[(year - self.start_year) as usize].contains(&date)
    }

    pub fn is_closing_day(&self, date: NaiveDate) -> bool {
        let year = date.year();
        assert!(year >= self.start_year && year <= self.end_year);

        self.yearly_closes[(year - self.start_year) as usize].contains(&date)
    }

    pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate {
        let mut next_date = date + Duration::days(1);
        while !self.is_trading_day(next_date) {
            next_date += Duration::days(1);
        }
        next_date
    }

    pub fn prev_trading_day(&self, date: NaiveDate) -> NaiveDate {
        let mut prev_date = date - Duration::days(1);
        while !self.is_trading_day(prev_date) {
            prev_date -= Duration::days(1);
        }
        prev_date
    }

    pub fn latest_trade_date(&self, asof: DateTime<Tz>) -> NaiveDate {
        let date = asof.date_naive();
        if !self.is_trading_day(date) {
            return self.prev_trading_day(date);
        }

        let (_, close_dt) = self.session(date);
        if asof >= close_dt {
            return date;
        }

        self.prev_trading_day(date)
    }

    pub fn session(&self, date: NaiveDate) -> (DateTime<Tz>, DateTime<Tz>) {
        let open_dt_local = date.and_time(chrono::NaiveTime::from_hms_opt(9, 30, 0).unwrap());
        let open_dt = New_York.from_local_datetime(&open_dt_local).unwrap();
        let close_dt = open_dt + if self.is_closing_day(date) {
            Duration::minutes(3*60+30) // 1:00 PM
        } else {
            Duration::minutes(6*60+30) // 4:00 PM
        };
        (open_dt, close_dt)
    }

    fn observed(date: NaiveDate) -> NaiveDate {
        let weekday = date.weekday();
        if weekday == Weekday::Sat {
            // Sat -> Fri
            date - Duration::days(1)
        } else if weekday == Weekday::Sun {
            // Sun -> Mon
            date + Duration::days(1)
        } else {
            date
        }
    }

    fn nth_weekday(year: i32, month: u32, weekday_num: u32, n: u32) -> NaiveDate {
        let first_day = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let first_weekday_num = first_day.weekday().num_days_from_monday() as i32;
        let target = weekday_num as i32;

        // use signed arithmetic to avoid underflow when target < first_weekday_num
        let days_to_nth_weekday = (n as i32 - 1) * 7 + (target - first_weekday_num).rem_euclid(7);
        first_day + Duration::days(days_to_nth_weekday as i64)
    }

    fn last_weekday(year: i32, month: u32, weekday_num: u32) -> NaiveDate {
        let last_day = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap() - Duration::days(1)
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap() - Duration::days(1)
        };
        let last_weekday_num = last_day.weekday().num_days_from_monday() as i32;
        let target = weekday_num as i32;
        let days_back = (last_weekday_num - target).rem_euclid(7) as i64;
        last_day - Duration::days(days_back)
    }

    fn easter(year: i32) -> NaiveDate {
        let a = year % 19;
        let b = year / 100;
        let c = year % 100;
        let d = b / 4;
        let e = b % 4;
        let f = (b + 8) / 25;
        let g = (b - f + 1) / 3;
        let h = (19 * a + b - d - g + 15) % 30;
        let i = c / 4;
        let k = c % 4;
        let l = (32 + 2 * e + 2 * i - h - k) % 7;
        let m = (a + 11 * h + 22 * l) / 451;
        let month = (h + l - 7 * m + 114) / 31;
        let day = ((h + l - 7 * m + 114) % 31) + 1;
        NaiveDate::from_ymd_opt(year, month as u32, day as u32).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, NaiveTime};

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn holidays_match_calendar_fixture() {
        let cal = USMarketCalendar::new(2024, 2025);

        let expected_2024 = vec![
            d("2024-01-01"),
            d("2024-01-15"),
            d("2024-02-19"),
            d("2024-03-29"),
            d("2024-05-27"),
            d("2024-06-19"),
            d("2024-07-04"),
            d("2024-09-02"),
            d("2024-11-28"),
            d("2024-12-25"),
        ];
        let mut got_2024 = cal.holidays[0].clone();
        got_2024.sort();
        assert_eq!(got_2024, expected_2024);

        let expected_2025 = vec![
            d("2025-01-01"),
            d("2025-01-20"),
            d("2025-02-17"),
            d("2025-04-18"),
            d("2025-05-26"),
            d("2025-06-19"),
            d("2025-07-04"),
            d("2025-09-01"),
            d("2025-11-27"),
            d("2025-12-25"),
        ];
        let mut got_2025 = cal.holidays[1].clone();
        got_2025.sort();
        assert_eq!(got_2025, expected_2025);
    }

    #[test]
    fn early_closes_match_calendar_fixture() {
        let cal = USMarketCalendar::new(2024, 2025);

        let expected_2024 = vec![d("2024-11-29"), d("2024-12-24")];
        let mut got_2024 = cal.yearly_closes[0].clone();
        got_2024.sort();
        assert_eq!(got_2024, expected_2024);

        let expected_2025 = vec![d("2025-11-28"), d("2025-12-24")];
        let mut got_2025 = cal.yearly_closes[1].clone();
        got_2025.sort();
        assert_eq!(got_2025, expected_2025);
    }

    #[test]
    fn session_times_align_with_calendar() {
        let cal = USMarketCalendar::new(2025, 2025);

        let (open_regular, close_regular) = cal.session(d("2025-06-03"));
        assert_eq!(open_regular.time(), NaiveTime::from_hms_opt(9, 30, 0).unwrap());
        assert_eq!(close_regular.time(), NaiveTime::from_hms_opt(16, 0, 0).unwrap());

        let (_open_early, close_early) = cal.session(d("2025-11-28"));
        assert_eq!(close_early.time(), NaiveTime::from_hms_opt(13, 0, 0).unwrap());
    }

    #[test]
    fn latest_trade_date_matches_logic() {
        let cal = USMarketCalendar::new(2025, 2025);

        // Holiday should roll back to the previous trading day (Fri before MLK).
        let mlk = New_York
            .with_ymd_and_hms(2025, 1, 20, 12, 0, 0)
            .unwrap();
        assert_eq!(cal.latest_trade_date(mlk), d("2025-01-17"));

        // After the close on a regular trading day should return the same date.
        let after_close = New_York
            .with_ymd_and_hms(2025, 6, 3, 17, 0, 0)
            .unwrap();
        assert_eq!(cal.latest_trade_date(after_close), d("2025-06-03"));

        // Before the open should roll back to the previous trading day.
        let pre_open = New_York
            .with_ymd_and_hms(2025, 6, 3, 8, 0, 0)
            .unwrap();
        assert_eq!(cal.latest_trade_date(pre_open), d("2025-06-02"));
    }

    #[test]
    fn trading_day_flags_and_navigation() {
        let cal = USMarketCalendar::new(2025, 2025);

        // weekend + holiday false; following Tuesday true
        assert!(!cal.is_trading_day(d("2025-01-18"))); // Sat
        assert!(!cal.is_trading_day(d("2025-01-19"))); // Sun
        assert!(!cal.is_trading_day(d("2025-01-20"))); // MLK
        assert!(cal.is_trading_day(d("2025-01-21")));  // Tue after MLK

        // navigation skips weekend + holiday
        assert_eq!(cal.next_trading_day(d("2025-01-17")), d("2025-01-21"));
        assert_eq!(cal.prev_trading_day(d("2025-01-21")), d("2025-01-17"));
    }
}
