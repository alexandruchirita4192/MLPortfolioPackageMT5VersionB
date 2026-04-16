#property strict
#property version   "1.00"
#property description "MT5 Portfolio EA Version B"
#property description "One executor per symbol, multiple model opinions aggregated before trading"

#include <Trade/Trade.mqh>

// BTC models
#resource "btc_linearsvc_calibrated.onnx" as uchar BtcLinearSvcModel[]
#resource "btc_logistic_regression.onnx" as uchar BtcLogRegModel[]

// XAG models
#resource "xag_mlp.onnx" as uchar XagMlpModel[]
#resource "xag_lightgbm.onnx" as uchar XagLgbmModel[]
#resource "xag_hgb.onnx" as uchar XagHgbModel[]
#resource "xag_linearsvc_calibrated.onnx" as uchar XagLinearSvcModel[]

const int FEATURE_COUNT = 13;
const int CLASS_COUNT   = 3;
const long EXT_INPUT_SHAPE[] = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[] = {1};
const long EXT_PROBA_SHAPE[] = {1, CLASS_COUNT};

enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };

struct SignalInfo
  {
   SignalDirection signal;
   double pSell;
   double pFlat;
   double pBuy;
   double bestDirectionProb;
   double probGap;
  };

struct SleeveState
  {
   bool enabled;
   string symbol;
   ENUM_TIMEFRAMES timeframe;

   double baseLots;
   bool useConfidenceSizing;
   double minLotMultiplier;
   double maxLotMultiplier;

   double entryProbThreshold;
   double minProbGap;
   bool useAtrStops;
   double stopAtrMultiple;
   double takeAtrMultiple;
   int maxBarsInTrade;
   bool closeOnOppositeSignal;
   bool allowLong;
   bool allowShort;

   bool useHourFilter;
   int hourStart;
   int hourEnd;

   bool useSpreadGuard;
   bool useAdaptiveSpreadGuard;
   double maxSpreadPoints;
   double maxSpreadPctOfPrice;
   double maxSpreadAtrFraction;
   double adaptiveSpreadSlack;

   bool useCooldownAfterClose;
   int cooldownBars;
   bool useDailyLossGuard;
   double dailyLossLimitMoney;
   bool dailyLossFlatOnTrigger;

   long magic;

   datetime lastBarTime;
   int barsInTrade;
   int cooldownRemaining;
   int lastHistoryDealsTotal;
   int dayKey;
   double dayClosedPnl;
   bool dailyLossGuardActive;
  };

// ------------------------------------------------------------
// BTC sleeve inputs
// ------------------------------------------------------------

input bool   InpEnableBTCUSD                     = true;
input string InpBtcSymbol                        = "BTCUSD";
input ENUM_TIMEFRAMES InpBtcTF                   = PERIOD_M15;

input double InpBtcBaseLots                      = 0.10;
input bool   InpBtcUseConfidenceSizing           = true;
input double InpBtcMinLotMultiplier              = 0.50;
input double InpBtcMaxLotMultiplier              = 1.50;

input double InpBtcEntryProbThreshold            = 0.60;
input double InpBtcMinProbGap                    = 0.20;
input bool   InpBtcUseAtrStops                   = true;
input double InpBtcStopAtrMultiple               = 1.00;
input double InpBtcTakeAtrMultiple               = 2.00;
input int    InpBtcMaxBarsInTrade                = 12;
input bool   InpBtcCloseOnOppositeSignal         = false;
input bool   InpBtcAllowLong                     = true;
input bool   InpBtcAllowShort                    = true;

input bool   InpBtcUseHourFilter                 = false;
input int    InpBtcHourStart                     = 0;
input int    InpBtcHourEnd                       = 23;

input bool   InpBtcUseSpreadGuard                = false;
input bool   InpBtcUseAdaptiveSpreadGuard        = true;
input double InpBtcMaxSpreadPoints               = 800.0;
input double InpBtcMaxSpreadPctOfPrice           = 0.0015;
input double InpBtcMaxSpreadAtrFraction          = 0.20;
input double InpBtcAdaptiveSpreadSlack           = 1.15;

input bool   InpBtcUseCooldownAfterClose         = false;
input int    InpBtcCooldownBars                  = 2;
input bool   InpBtcUseDailyLossGuard             = false;
input double InpBtcDailyLossLimitMoney           = 300.0;
input bool   InpBtcDailyLossFlatOnTrigger        = true;

input long   InpBtcMagic                         = 26042161;

// BTC model aggregation
input bool   InpBtcUseLinearSvc                  = true;
input bool   InpBtcUseLogisticRegression         = true;
input double InpBtcLinearSvcWeight               = 0.70;
input double InpBtcLogRegWeight                  = 0.30;

// ------------------------------------------------------------
// XAG sleeve inputs
// ------------------------------------------------------------

input bool   InpEnableXAGUSD                     = true;
input string InpXagSymbol                        = "XAGUSD";
input ENUM_TIMEFRAMES InpXagTF                   = PERIOD_M15;

input double InpXagBaseLots                      = 0.10;
input bool   InpXagUseConfidenceSizing           = true;
input double InpXagMinLotMultiplier              = 0.50;
input double InpXagMaxLotMultiplier              = 1.50;

input double InpXagEntryProbThreshold            = 0.60;
input double InpXagMinProbGap                    = 0.00;
input bool   InpXagUseAtrStops                   = true;
input double InpXagStopAtrMultiple               = 1.00;
input double InpXagTakeAtrMultiple               = 2.75;
input int    InpXagMaxBarsInTrade                = 8;
input bool   InpXagCloseOnOppositeSignal         = false;
input bool   InpXagAllowLong                     = true;
input bool   InpXagAllowShort                    = true;

input bool   InpXagUseHourFilter                 = false;
input int    InpXagHourStart                     = 0;
input int    InpXagHourEnd                       = 23;

input bool   InpXagUseSpreadGuard                = false;
input bool   InpXagUseAdaptiveSpreadGuard        = true;
input double InpXagMaxSpreadPoints               = 150.0;
input double InpXagMaxSpreadPctOfPrice           = 0.0010;
input double InpXagMaxSpreadAtrFraction          = 0.20;
input double InpXagAdaptiveSpreadSlack           = 1.10;

input bool   InpXagUseCooldownAfterClose         = false;
input int    InpXagCooldownBars                  = 2;
input bool   InpXagUseDailyLossGuard             = false;
input double InpXagDailyLossLimitMoney           = 150.0;
input bool   InpXagDailyLossFlatOnTrigger        = true;

input long   InpXagMagic                         = 26042162;

// XAG internal ensemble weights
input double InpXagMlpWeight                     = 0.25;
input double InpXagLgbmWeight                    = 0.25;
input double InpXagHgbWeight                     = 0.50;

// XAG top-level aggregation
input bool   InpXagUseEnsembleGroup              = true;
input bool   InpXagUseLinearSvc                  = false;
input double InpXagEnsembleGroupWeight           = 0.80;
input double InpXagLinearSvcWeight               = 0.20;

// Portfolio-level guard
input bool   InpUsePortfolioDailyLossGuard       = false;
input double InpPortfolioDailyLossLimitMoney     = 500.0;
input bool   InpPortfolioFlatOnTrigger           = true;

input bool   InpLog                              = false;
input bool   InpDebugLog                         = false;

// ------------------------------------------------------------
// Global state
// ------------------------------------------------------------

CTrade trade;

SleeveState g_btc;
SleeveState g_xag;

long g_btc_linearsvc_handle = INVALID_HANDLE;
long g_btc_logreg_handle    = INVALID_HANDLE;

long g_xag_mlp_handle       = INVALID_HANDLE;
long g_xag_lgbm_handle      = INVALID_HANDLE;
long g_xag_hgb_handle       = INVALID_HANDLE;
long g_xag_linearsvc_handle = INVALID_HANDLE;

double g_btc_w_linearsvc = 0.0;
double g_btc_w_logreg    = 0.0;

double g_xag_w_mlp = 0.0;
double g_xag_w_lgbm = 0.0;
double g_xag_w_hgb = 0.0;

double g_xag_w_ensemble_group = 0.0;
double g_xag_w_linearsvc      = 0.0;

int g_portfolio_day_key = -1;
double g_portfolio_day_closed_pnl = 0.0;
bool g_portfolio_daily_loss_active = false;

// ------------------------------------------------------------
// Generic helpers
// ------------------------------------------------------------

int DayKey(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
  }

void ResetPortfolioDailyStateIfNeeded()
  {
   int today_key = DayKey(TimeCurrent());
   if(g_portfolio_day_key != today_key)
     {
      g_portfolio_day_key = today_key;
      g_portfolio_day_closed_pnl = 0.0;
      g_portfolio_daily_loss_active = false;
     }
  }

double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }

double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }

double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }

bool IsHourInRange(int hour_value, int start_hour, int end_hour)
  {
   if(start_hour < 0 || start_hour > 23 || end_hour < 0 || end_hour > 23)
      return false;

   if(start_hour <= end_hour)
      return (hour_value >= start_hour && hour_value <= end_hour);

   return (hour_value >= start_hour || hour_value <= end_hour);
  }

int CurrentServerHour()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return dt.hour;
  }

double NormalizeVolumeToSymbol(const string symbol, double requested_lots)
  {
   double vol_min  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vol_max  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vol_step <= 0.0)
      vol_step = vol_min;

   double lots = MathMax(vol_min, MathMin(vol_max, requested_lots));
   lots = MathFloor(lots / vol_step) * vol_step;

   int digits = 2;
   if(vol_step > 0.0)
     {
      double tmp = vol_step;
      digits = 0;
      while(digits < 8 && MathRound(tmp) != tmp)
        {
         tmp *= 10.0;
         digits++;
        }
     }

   lots = NormalizeDouble(lots, digits);
   return MathMax(vol_min, MathMin(vol_max, lots));
  }

bool NormalizeTwoWeights(bool useA, double inA, bool useB, double inB, double &outA, double &outB)
  {
   outA = 0.0;
   outB = 0.0;

   double a = useA ? MathMax(0.0, inA) : 0.0;
   double b = useB ? MathMax(0.0, inB) : 0.0;
   double s = a + b;

   if(s <= 0.0)
      return false;

   outA = a / s;
   outB = b / s;
   return true;
  }

bool NormalizeThreeWeights(double inA, double inB, double inC, double &outA, double &outB, double &outC)
  {
   double a = MathMax(0.0, inA);
   double b = MathMax(0.0, inB);
   double c = MathMax(0.0, inC);
   double s = a + b + c;

   if(s <= 0.0)
      return false;

   outA = a / s;
   outB = b / s;
   outC = c / s;
   return true;
  }

bool InitSingleModel(long &handle_ref, const uchar &buffer[])
  {
   handle_ref = OnnxCreateFromBuffer(buffer, ONNX_DEFAULT);
   if(handle_ref == INVALID_HANDLE)
      return false;

   if(!OnnxSetInputShape(handle_ref, 0, EXT_INPUT_SHAPE))
      return false;
   if(!OnnxSetOutputShape(handle_ref, 0, EXT_LABEL_SHAPE))
      return false;
   if(!OnnxSetOutputShape(handle_ref, 1, EXT_PROBA_SHAPE))
      return false;

   return true;
  }

void InitSleeveDefaults()
  {
   g_btc.enabled = InpEnableBTCUSD;
   g_btc.symbol = InpBtcSymbol;
   g_btc.timeframe = InpBtcTF;
   g_btc.baseLots = InpBtcBaseLots;
   g_btc.useConfidenceSizing = InpBtcUseConfidenceSizing;
   g_btc.minLotMultiplier = InpBtcMinLotMultiplier;
   g_btc.maxLotMultiplier = InpBtcMaxLotMultiplier;
   g_btc.entryProbThreshold = InpBtcEntryProbThreshold;
   g_btc.minProbGap = InpBtcMinProbGap;
   g_btc.useAtrStops = InpBtcUseAtrStops;
   g_btc.stopAtrMultiple = InpBtcStopAtrMultiple;
   g_btc.takeAtrMultiple = InpBtcTakeAtrMultiple;
   g_btc.maxBarsInTrade = InpBtcMaxBarsInTrade;
   g_btc.closeOnOppositeSignal = InpBtcCloseOnOppositeSignal;
   g_btc.allowLong = InpBtcAllowLong;
   g_btc.allowShort = InpBtcAllowShort;
   g_btc.useHourFilter = InpBtcUseHourFilter;
   g_btc.hourStart = InpBtcHourStart;
   g_btc.hourEnd = InpBtcHourEnd;
   g_btc.useSpreadGuard = InpBtcUseSpreadGuard;
   g_btc.useAdaptiveSpreadGuard = InpBtcUseAdaptiveSpreadGuard;
   g_btc.maxSpreadPoints = InpBtcMaxSpreadPoints;
   g_btc.maxSpreadPctOfPrice = InpBtcMaxSpreadPctOfPrice;
   g_btc.maxSpreadAtrFraction = InpBtcMaxSpreadAtrFraction;
   g_btc.adaptiveSpreadSlack = InpBtcAdaptiveSpreadSlack;
   g_btc.useCooldownAfterClose = InpBtcUseCooldownAfterClose;
   g_btc.cooldownBars = InpBtcCooldownBars;
   g_btc.useDailyLossGuard = InpBtcUseDailyLossGuard;
   g_btc.dailyLossLimitMoney = InpBtcDailyLossLimitMoney;
   g_btc.dailyLossFlatOnTrigger = InpBtcDailyLossFlatOnTrigger;
   g_btc.magic = InpBtcMagic;
   g_btc.lastBarTime = 0;
   g_btc.barsInTrade = 0;
   g_btc.cooldownRemaining = 0;
   g_btc.lastHistoryDealsTotal = 0;
   g_btc.dayKey = -1;
   g_btc.dayClosedPnl = 0.0;
   g_btc.dailyLossGuardActive = false;

   g_xag.enabled = InpEnableXAGUSD;
   g_xag.symbol = InpXagSymbol;
   g_xag.timeframe = InpXagTF;
   g_xag.baseLots = InpXagBaseLots;
   g_xag.useConfidenceSizing = InpXagUseConfidenceSizing;
   g_xag.minLotMultiplier = InpXagMinLotMultiplier;
   g_xag.maxLotMultiplier = InpXagMaxLotMultiplier;
   g_xag.entryProbThreshold = InpXagEntryProbThreshold;
   g_xag.minProbGap = InpXagMinProbGap;
   g_xag.useAtrStops = InpXagUseAtrStops;
   g_xag.stopAtrMultiple = InpXagStopAtrMultiple;
   g_xag.takeAtrMultiple = InpXagTakeAtrMultiple;
   g_xag.maxBarsInTrade = InpXagMaxBarsInTrade;
   g_xag.closeOnOppositeSignal = InpXagCloseOnOppositeSignal;
   g_xag.allowLong = InpXagAllowLong;
   g_xag.allowShort = InpXagAllowShort;
   g_xag.useHourFilter = InpXagUseHourFilter;
   g_xag.hourStart = InpXagHourStart;
   g_xag.hourEnd = InpXagHourEnd;
   g_xag.useSpreadGuard = InpXagUseSpreadGuard;
   g_xag.useAdaptiveSpreadGuard = InpXagUseAdaptiveSpreadGuard;
   g_xag.maxSpreadPoints = InpXagMaxSpreadPoints;
   g_xag.maxSpreadPctOfPrice = InpXagMaxSpreadPctOfPrice;
   g_xag.maxSpreadAtrFraction = InpXagMaxSpreadAtrFraction;
   g_xag.adaptiveSpreadSlack = InpXagAdaptiveSpreadSlack;
   g_xag.useCooldownAfterClose = InpXagUseCooldownAfterClose;
   g_xag.cooldownBars = InpXagCooldownBars;
   g_xag.useDailyLossGuard = InpXagUseDailyLossGuard;
   g_xag.dailyLossLimitMoney = InpXagDailyLossLimitMoney;
   g_xag.dailyLossFlatOnTrigger = InpXagDailyLossFlatOnTrigger;
   g_xag.magic = InpXagMagic;
   g_xag.lastBarTime = 0;
   g_xag.barsInTrade = 0;
   g_xag.cooldownRemaining = 0;
   g_xag.lastHistoryDealsTotal = 0;
   g_xag.dayKey = -1;
   g_xag.dayClosedPnl = 0.0;
   g_xag.dailyLossGuardActive = false;
  }

bool InitAggregationWeights()
  {
   if(!NormalizeTwoWeights(
         InpBtcUseLinearSvc, InpBtcLinearSvcWeight,
         InpBtcUseLogisticRegression, InpBtcLogRegWeight,
         g_btc_w_linearsvc, g_btc_w_logreg))
      return false;

   if(!NormalizeThreeWeights(
         InpXagMlpWeight, InpXagLgbmWeight, InpXagHgbWeight,
         g_xag_w_mlp, g_xag_w_lgbm, g_xag_w_hgb))
      return false;

   if(!NormalizeTwoWeights(
         InpXagUseEnsembleGroup, InpXagEnsembleGroupWeight,
         InpXagUseLinearSvc, InpXagLinearSvcWeight,
         g_xag_w_ensemble_group, g_xag_w_linearsvc))
      return false;

   return true;
  }

bool InitAllModels()
  {
   if(g_btc.enabled)
     {
      if(InpBtcUseLinearSvc)
         if(!InitSingleModel(g_btc_linearsvc_handle, BtcLinearSvcModel))
            return false;

      if(InpBtcUseLogisticRegression)
         if(!InitSingleModel(g_btc_logreg_handle, BtcLogRegModel))
            return false;
     }

   if(g_xag.enabled)
     {
      if(InpXagUseEnsembleGroup)
        {
         if(!InitSingleModel(g_xag_mlp_handle, XagMlpModel))
            return false;
         if(!InitSingleModel(g_xag_lgbm_handle, XagLgbmModel))
            return false;
         if(!InitSingleModel(g_xag_hgb_handle, XagHgbModel))
            return false;
        }

      if(InpXagUseLinearSvc)
         if(!InitSingleModel(g_xag_linearsvc_handle, XagLinearSvcModel))
            return false;
     }

   return true;
  }

bool HasOpenPositionForSleeve(const SleeveState &s, long &pos_type, double &pos_price)
  {
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      string psymbol = PositionGetString(POSITION_SYMBOL);
      long pmagic = (long)PositionGetInteger(POSITION_MAGIC);

      if(psymbol == s.symbol && pmagic == s.magic)
        {
         pos_type = (long)PositionGetInteger(POSITION_TYPE);
         pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
         return true;
        }
     }
   return false;
  }

void CloseOpenPositionForSleeve(const SleeveState &s)
  {
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      string psymbol = PositionGetString(POSITION_SYMBOL);
      long pmagic = (long)PositionGetInteger(POSITION_MAGIC);

      if(psymbol == s.symbol && pmagic == s.magic)
        {
         trade.SetExpertMagicNumber(s.magic);
         trade.PositionClose(ticket);
        }
     }
  }

void ResetSleeveDailyStateIfNeeded(SleeveState &s)
  {
   int today_key = DayKey(TimeCurrent());
   if(s.dayKey != today_key)
     {
      s.dayKey = today_key;
      s.dayClosedPnl = 0.0;
      s.dailyLossGuardActive = false;
     }
  }

void RefreshClosedDealState(SleeveState &s)
  {
   ResetSleeveDailyStateIfNeeded(s);
   ResetPortfolioDailyStateIfNeeded();

   if(!HistorySelect(0, TimeCurrent()))
      return;

   int total = HistoryDealsTotal();
   if(total <= s.lastHistoryDealsTotal)
      return;

   for(int i = s.lastHistoryDealsTotal; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long magic    = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      long entry    = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(symbol != s.symbol || magic != s.magic || entry != DEAL_ENTRY_OUT)
         continue;

      double profit     = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double swap       = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      double net = profit + swap + commission;

      datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      int deal_day_key = DayKey(deal_time);

      if(deal_day_key == s.dayKey)
         s.dayClosedPnl += net;
      if(deal_day_key == g_portfolio_day_key)
         g_portfolio_day_closed_pnl += net;

      if(s.useCooldownAfterClose)
         s.cooldownRemaining = s.cooldownBars;
     }

   s.lastHistoryDealsTotal = total;

   if(s.useDailyLossGuard && !s.dailyLossGuardActive && s.dayClosedPnl <= -MathAbs(s.dailyLossLimitMoney))
     {
      s.dailyLossGuardActive = true;
      if(s.dailyLossFlatOnTrigger)
         CloseOpenPositionForSleeve(s);
     }

   if(InpUsePortfolioDailyLossGuard && !g_portfolio_daily_loss_active &&
      g_portfolio_day_closed_pnl <= -MathAbs(InpPortfolioDailyLossLimitMoney))
     {
      g_portfolio_daily_loss_active = true;
      if(InpPortfolioFlatOnTrigger)
        {
         CloseOpenPositionForSleeve(g_btc);
         CloseOpenPositionForSleeve(g_xag);
        }
     }
  }

void DecrementCooldown(SleeveState &s)
  {
   if(s.cooldownRemaining > 0)
      s.cooldownRemaining--;
  }

bool IsNewBarForSleeve(SleeveState &s)
  {
   datetime current_bar_time = iTime(s.symbol, s.timeframe, 0);
   if(current_bar_time == 0)
      return false;

   if(s.lastBarTime == 0)
     {
      s.lastBarTime = current_bar_time;
      return false;
     }

   if(current_bar_time != s.lastBarTime)
     {
      s.lastBarTime = current_bar_time;
      return true;
     }

   return false;
  }

bool BuildFeatureVectorForSleeve(const SleeveState &s, matrixf &features, double &atr14_raw)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(s.symbol, s.timeframe, 0, 80, rates) < 40)
      return false;

   double closes[], opens[];
   ArrayResize(closes, ArraySize(rates));
   ArrayResize(opens, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(opens, true);

   for(int i = 0; i < ArraySize(rates); i++)
     {
      closes[i] = rates[i].close;
      opens[i]  = rates[i].open;
     }

   int sh = 1;
   double eps = 1e-12;
   double c = closes[sh];
   double o = opens[sh];
   double h = rates[sh].high;
   double l = rates[sh].low;

   double ret_1  = (closes[sh] / (closes[sh + 1] + eps)) - 1.0;
   double ret_3  = (closes[sh] / (closes[sh + 3] + eps)) - 1.0;
   double ret_5  = (closes[sh] / (closes[sh + 5] + eps)) - 1.0;
   double ret_10 = (closes[sh] / (closes[sh + 10] + eps)) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[sh + i] / (closes[sh + i + 1] + eps)) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);
   double vol_ratio_10_20 = (vol_10 / (vol_20 + eps)) - 1.0;

   double sma_10 = Mean(closes, sh, 10);
   double sma_20 = Mean(closes, sh, 20);
   if(sma_10 == 0.0 || sma_20 == 0.0)
      return false;

   double dist_sma_10 = (c / (sma_10 + eps)) - 1.0;
   double dist_sma_20 = (c / (sma_20 + eps)) - 1.0;

   double mean_20 = Mean(closes, sh, 20);
   double std_20  = StdDev(closes, sh, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (c - mean_20) / std_20;

   atr14_raw = CalcATR(rates, sh, 14);
   double atr_pct_14 = atr14_raw / (c + eps);
   double range_pct_1 = (h - l) / (c + eps);
   double body_pct_1 = (c - o) / (o + eps);

   features.Resize(1, FEATURE_COUNT);
   features[0][0]  = (float)ret_1;
   features[0][1]  = (float)ret_3;
   features[0][2]  = (float)ret_5;
   features[0][3]  = (float)ret_10;
   features[0][4]  = (float)vol_10;
   features[0][5]  = (float)vol_20;
   features[0][6]  = (float)vol_ratio_10_20;
   features[0][7]  = (float)dist_sma_10;
   features[0][8]  = (float)dist_sma_20;
   features[0][9]  = (float)zscore_20;
   features[0][10] = (float)atr_pct_14;
   features[0][11] = (float)range_pct_1;
   features[0][12] = (float)body_pct_1;

   return true;
  }

bool RunSingleModel(long model_handle, const matrixf &x, double &pSell, double &pFlat, double &pBuy)
  {
   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(model_handle, 0, x, predicted_label, probs))
      return false;

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy  = probs[0][2];
   return true;
  }

bool PredictBtcAggregatedProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14_raw)
  {
   matrixf x;
   if(!BuildFeatureVectorForSleeve(g_btc, x, atr14_raw))
      return false;

   double finalSell = 0.0, finalFlat = 0.0, finalBuy = 0.0;

   if(InpBtcUseLinearSvc)
     {
      double s1, f1, b1;
      if(!RunSingleModel(g_btc_linearsvc_handle, x, s1, f1, b1))
         return false;
      finalSell += g_btc_w_linearsvc * s1;
      finalFlat += g_btc_w_linearsvc * f1;
      finalBuy  += g_btc_w_linearsvc * b1;
     }

   if(InpBtcUseLogisticRegression)
     {
      double s2, f2, b2;
      if(!RunSingleModel(g_btc_logreg_handle, x, s2, f2, b2))
         return false;
      finalSell += g_btc_w_logreg * s2;
      finalFlat += g_btc_w_logreg * f2;
      finalBuy  += g_btc_w_logreg * b2;
     }

   pSell = finalSell;
   pFlat = finalFlat;
   pBuy  = finalBuy;
   return true;
  }

bool PredictXagEnsembleGroup(double &pSell, double &pFlat, double &pBuy, const matrixf &x)
  {
   double s1, f1, b1;
   double s2, f2, b2;
   double s3, f3, b3;

   if(!RunSingleModel(g_xag_mlp_handle, x, s1, f1, b1))
      return false;
   if(!RunSingleModel(g_xag_lgbm_handle, x, s2, f2, b2))
      return false;
   if(!RunSingleModel(g_xag_hgb_handle, x, s3, f3, b3))
      return false;

   pSell = g_xag_w_mlp * s1 + g_xag_w_lgbm * s2 + g_xag_w_hgb * s3;
   pFlat = g_xag_w_mlp * f1 + g_xag_w_lgbm * f2 + g_xag_w_hgb * f3;
   pBuy  = g_xag_w_mlp * b1 + g_xag_w_lgbm * b2 + g_xag_w_hgb * b3;
   return true;
  }

bool PredictXagAggregatedProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14_raw)
  {
   matrixf x;
   if(!BuildFeatureVectorForSleeve(g_xag, x, atr14_raw))
      return false;

   double finalSell = 0.0, finalFlat = 0.0, finalBuy = 0.0;

   if(InpXagUseEnsembleGroup)
     {
      double se, fe, be;
      if(!PredictXagEnsembleGroup(se, fe, be, x))
         return false;
      finalSell += g_xag_w_ensemble_group * se;
      finalFlat += g_xag_w_ensemble_group * fe;
      finalBuy  += g_xag_w_ensemble_group * be;
     }

   if(InpXagUseLinearSvc)
     {
      double sls, fls, bls;
      if(!RunSingleModel(g_xag_linearsvc_handle, x, sls, fls, bls))
         return false;
      finalSell += g_xag_w_linearsvc * sls;
      finalFlat += g_xag_w_linearsvc * fls;
      finalBuy  += g_xag_w_linearsvc * bls;
     }

   pSell = finalSell;
   pFlat = finalFlat;
   pBuy  = finalBuy;
   return true;
  }

SignalInfo BuildSignalInfoForSleeve(const SleeveState &s, double pSell, double pFlat, double pBuy)
  {
   SignalInfo info;
   info.signal = SIGNAL_FLAT;
   info.pSell = pSell;
   info.pFlat = pFlat;
   info.pBuy  = pBuy;
   info.bestDirectionProb = MathMax(pSell, pBuy);
   info.probGap = 0.0;

   double best = pFlat;
   double second = -1.0;
   SignalDirection signal = SIGNAL_FLAT;

   if(pBuy >= pSell && pBuy > best)
     {
      second = MathMax(best, pSell);
      best = pBuy;
      signal = SIGNAL_BUY;
     }
   else if(pSell > pBuy && pSell > best)
     {
      second = MathMax(best, pBuy);
      best = pSell;
      signal = SIGNAL_SELL;
     }
   else
     {
      second = MathMax(pBuy, pSell);
      signal = SIGNAL_FLAT;
     }

   info.bestDirectionProb = best;
   info.probGap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!s.allowLong || pBuy < s.entryProbThreshold || info.probGap < s.minProbGap)
         info.signal = SIGNAL_FLAT;
      else
         info.signal = SIGNAL_BUY;
      return info;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!s.allowShort || pSell < s.entryProbThreshold || info.probGap < s.minProbGap)
         info.signal = SIGNAL_FLAT;
      else
         info.signal = SIGNAL_SELL;
      return info;
     }

   info.signal = SIGNAL_FLAT;
   return info;
  }

double GetCurrentSpreadPointsForSleeve(const SleeveState &s)
  {
   double ask = SymbolInfoDouble(s.symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(s.symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(s.symbol, SYMBOL_POINT);
   if(point <= 0.0)
      return 0.0;
   return (ask - bid) / point;
  }

double GetCurrentSpreadPriceForSleeve(const SleeveState &s)
  {
   double ask = SymbolInfoDouble(s.symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(s.symbol, SYMBOL_BID);
   return ask - bid;
  }

bool SpreadAllowsForSleeve(const SleeveState &s, double atr14_raw)
  {
   if(!s.useSpreadGuard)
      return true;

   double ask = SymbolInfoDouble(s.symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(s.symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;
   double spread_points = GetCurrentSpreadPointsForSleeve(s);
   double spread_price = GetCurrentSpreadPriceForSleeve(s);

   if(mid <= 0.0)
      return false;

   if(!s.useAdaptiveSpreadGuard)
      return (spread_points <= s.maxSpreadPoints);

   double point = SymbolInfoDouble(s.symbol, SYMBOL_POINT);
   double max_from_points = s.maxSpreadPoints * point;
   double max_from_pct = mid * MathMax(0.0, s.maxSpreadPctOfPrice);
   double max_from_atr = atr14_raw * MathMax(0.0, s.maxSpreadAtrFraction);

   double final_limit = MathMin(max_from_points, MathMin(max_from_pct, max_from_atr));
   final_limit *= MathMax(1.0, s.adaptiveSpreadSlack);

   return (spread_price <= final_limit);
  }

bool EntryGuardsAllowForSleeve(const SleeveState &s, double atr14_raw)
  {
   if(InpUsePortfolioDailyLossGuard && g_portfolio_daily_loss_active)
      return false;
   if(s.useDailyLossGuard && s.dailyLossGuardActive)
      return false;
   if(s.useCooldownAfterClose && s.cooldownRemaining > 0)
      return false;

   if(s.useHourFilter)
     {
      int h = CurrentServerHour();
      if(!IsHourInRange(h, s.hourStart, s.hourEnd))
         return false;
     }

   if(!SpreadAllowsForSleeve(s, atr14_raw))
      return false;

   return true;
  }

double ComputeLotSizeForSleeve(const SleeveState &s, const SignalInfo &info)
  {
   double lots = s.baseLots;

   if(!s.useConfidenceSizing)
      return NormalizeVolumeToSymbol(s.symbol, lots);

   double strength_prob = MathMax(0.0, info.bestDirectionProb - s.entryProbThreshold);
   double span_prob = MathMax(1e-8, 1.0 - s.entryProbThreshold);
   double prob_score = MathMin(1.0, strength_prob / span_prob);

   double gap_score = 0.0;
   if(s.minProbGap <= 0.0)
      gap_score = MathMin(1.0, info.probGap / 0.25);
   else
      gap_score = MathMin(1.0, MathMax(0.0, info.probGap - s.minProbGap) / MathMax(1e-8, 0.30 - s.minProbGap));

   double blended = 0.70 * prob_score + 0.30 * gap_score;
   blended = MathMax(0.0, MathMin(1.0, blended));

   double mult = s.minLotMultiplier + (s.maxLotMultiplier - s.minLotMultiplier) * blended;
   lots *= mult;

   return NormalizeVolumeToSymbol(s.symbol, lots);
  }

void OpenTradeForSleeve(SleeveState &s, const SignalInfo &info, double atr14_raw)
  {
   double lots = ComputeLotSizeForSleeve(s, info);

   double ask = SymbolInfoDouble(s.symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(s.symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(s.symbol, SYMBOL_POINT);

   double min_stop = (double)SymbolInfoInteger(s.symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14_raw * s.stopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14_raw * s.takeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(s.magic);
   trade.SetDeviationInPoints(20);

   if(info.signal == SIGNAL_BUY)
     {
      if(s.useAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      if(trade.Buy(lots, s.symbol, ask, sl, tp, "Portfolio VB buy"))
         s.barsInTrade = 0;
     }
   else if(info.signal == SIGNAL_SELL)
     {
      if(s.useAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      if(trade.Sell(lots, s.symbol, bid, sl, tp, "Portfolio VB sell"))
         s.barsInTrade = 0;
     }
  }

void ManageExistingPositionForSleeve(SleeveState &s, const SignalInfo &info)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPositionForSleeve(s, pos_type, pos_price))
      return;

   s.barsInTrade++;
   bool should_close = false;

   if(s.closeOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && info.signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && info.signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && s.barsInTrade >= s.maxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPositionForSleeve(s);
  }

void ProcessBtcSleeve()
  {
   if(!g_btc.enabled)
      return;

   if(!IsNewBarForSleeve(g_btc))
      return;

   RefreshClosedDealState(g_btc);
   DecrementCooldown(g_btc);

   double pSell = 0.0, pFlat = 0.0, pBuy = 0.0, atr14_raw = 0.0;
   if(!PredictBtcAggregatedProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   SignalInfo info = BuildSignalInfoForSleeve(g_btc, pSell, pFlat, pBuy);

   if(InpDebugLog && InpLog)
      PrintFormat("BTC agg probs sell=%.4f flat=%.4f buy=%.4f best=%.4f gap=%.4f signal=%d",
                  info.pSell, info.pFlat, info.pBuy, info.bestDirectionProb, info.probGap, info.signal);

   ManageExistingPositionForSleeve(g_btc, info);

   long pos_type;
   double pos_price;
   if(HasOpenPositionForSleeve(g_btc, pos_type, pos_price))
      return;

   if(info.signal == SIGNAL_FLAT)
      return;

   if(!EntryGuardsAllowForSleeve(g_btc, atr14_raw))
      return;

   OpenTradeForSleeve(g_btc, info, atr14_raw);
  }

void ProcessXagSleeve()
  {
   if(!g_xag.enabled)
      return;

   if(!IsNewBarForSleeve(g_xag))
      return;

   RefreshClosedDealState(g_xag);
   DecrementCooldown(g_xag);

   double pSell = 0.0, pFlat = 0.0, pBuy = 0.0, atr14_raw = 0.0;
   if(!PredictXagAggregatedProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   SignalInfo info = BuildSignalInfoForSleeve(g_xag, pSell, pFlat, pBuy);

   if(InpDebugLog && InpLog)
      PrintFormat("XAG agg probs sell=%.4f flat=%.4f buy=%.4f best=%.4f gap=%.4f signal=%d",
                  info.pSell, info.pFlat, info.pBuy, info.bestDirectionProb, info.probGap, info.signal);

   ManageExistingPositionForSleeve(g_xag, info);

   long pos_type;
   double pos_price;
   if(HasOpenPositionForSleeve(g_xag, pos_type, pos_price))
      return;

   if(info.signal == SIGNAL_FLAT)
      return;

   if(!EntryGuardsAllowForSleeve(g_xag, atr14_raw))
      return;

   OpenTradeForSleeve(g_xag, info, atr14_raw);
  }

int OnInit()
  {
   InitSleeveDefaults();
   ResetPortfolioDailyStateIfNeeded();

   if(!InitAggregationWeights())
      return INIT_PARAMETERS_INCORRECT;

   if(!InitAllModels())
      return INIT_FAILED;

   if(HistorySelect(0, TimeCurrent()))
     {
      int hist = HistoryDealsTotal();
      g_btc.lastHistoryDealsTotal = hist;
      g_xag.lastHistoryDealsTotal = hist;
     }

   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(g_btc_linearsvc_handle != INVALID_HANDLE) OnnxRelease(g_btc_linearsvc_handle);
   if(g_btc_logreg_handle    != INVALID_HANDLE) OnnxRelease(g_btc_logreg_handle);

   if(g_xag_mlp_handle       != INVALID_HANDLE) OnnxRelease(g_xag_mlp_handle);
   if(g_xag_lgbm_handle      != INVALID_HANDLE) OnnxRelease(g_xag_lgbm_handle);
   if(g_xag_hgb_handle       != INVALID_HANDLE) OnnxRelease(g_xag_hgb_handle);
   if(g_xag_linearsvc_handle != INVALID_HANDLE) OnnxRelease(g_xag_linearsvc_handle);

   g_btc_linearsvc_handle = INVALID_HANDLE;
   g_btc_logreg_handle    = INVALID_HANDLE;
   g_xag_mlp_handle       = INVALID_HANDLE;
   g_xag_lgbm_handle      = INVALID_HANDLE;
   g_xag_hgb_handle       = INVALID_HANDLE;
   g_xag_linearsvc_handle = INVALID_HANDLE;
  }

void OnTick()
  {
   ProcessBtcSleeve();
   ProcessXagSleeve();
  }

double OnTester() {
  double profit = TesterStatistics(STAT_PROFIT);
  double pf = TesterStatistics(STAT_PROFIT_FACTOR);
  double recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
  double dd_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
  double trades = TesterStatistics(STAT_TRADES);

  // Penalty if there are too few transactions
  double trade_penalty = 1.0;
  if (trades < 20)
    trade_penalty = 0.25;
  else if (trades < 50)
    trade_penalty = 0.60;

  // Robust score, not only brut profit
  double score = 0.0;

  if (dd_percent >= 0.0)
    score =
        (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) /
        (1.0 + dd_percent);

  return score;
}
